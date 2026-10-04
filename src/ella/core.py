from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import torch

from .lora_utils import (
    _active_adapter_name,
    _normalize_module_name,
    collect_lora_deltas,
    iter_lora_deltas,
)


_SUBSPACE_EIGENVALUE_RTOL = 1e-4
_PAST_SUM_DTYPE = torch.float16


@dataclass
class ELLAState:
    """Stores all past LoRA factors per module.

    New states contain ``list[(B, A)]`` for each module.  A tensor is also
    accepted for backwards compatibility with states written by older
    versions, where it represented the already-summed ``W_past``.
    """

    past: Dict[str, object] = field(default_factory=dict)
    past_sum: Dict[str, torch.Tensor] = field(default_factory=dict)
    _subspace_cache: Dict[
        tuple[str, str],
        tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ] = field(default_factory=dict, init=False, repr=False)
    # Device/dtype-specific copies of ``past_sum``.  The serialized state
    # remains CPU-backed, while training reuses these copies between steps.
    _past_tensor_cache: Dict[tuple[str, str, str], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )
    _past_factor_cache: Dict[
        tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize legacy dense history while keeping factor history compact."""
        self.past_sum = {
            key: value.to(dtype=_PAST_SUM_DTYPE)
            for key, value in self.past_sum.items()
        }

    def save(self, path: str | Path) -> None:
        payload = {"past": {}, "past_sum": {}}
        for key, value in self.past.items():
            if torch.is_tensor(value):
                payload["past"][key] = value.detach().cpu()
            else:
                payload["past"][key] = [
                    (b.detach().cpu(), a.detach().cpu()) for b, a in value  # type: ignore[union-attr]
                ]
        payload["past_sum"] = {
            k: v.detach().to(device="cpu", dtype=_PAST_SUM_DTYPE)
            for k, v in self.past_sum.items()
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str | None = None) -> "ELLAState":
        payload = torch.load(path, map_location=device if device is not None else "cpu")
        if not isinstance(payload, dict):
            raise TypeError("Expected dict payload for ELLA state.")
        # Older checkpoints were the flat ``{name: summed_tensor}`` mapping.
        if "past" in payload and isinstance(payload["past"], dict):
            raw_past = payload["past"]
            raw_sum = payload.get("past_sum", {})
        else:
            raw_past, raw_sum = payload, {}
        out: Dict[str, object] = {}
        for key, value in raw_past.items():
            if torch.is_tensor(value):
                out[key] = value
                continue
            if isinstance(value, (list, tuple)):
                factors = []
                for factor in value:
                    if not isinstance(factor, (list, tuple)) or len(factor) != 2:
                        raise TypeError(f"ELLA state key '{key}' has invalid factor entry.")
                    b, a = factor
                    if not torch.is_tensor(b) or not torch.is_tensor(a):
                        raise TypeError(f"ELLA state key '{key}' has non-tensor factors.")
                    factors.append((b, a))
                out[key] = factors
                continue
            raise TypeError(f"ELLA state key '{key}' is not a tensor or factor list.")
        sums: Dict[str, torch.Tensor] = {}
        if isinstance(raw_sum, dict):
            for key, value in raw_sum.items():
                if not torch.is_tensor(value):
                    raise TypeError(f"ELLA past_sum key '{key}' is not a tensor.")
                sums[key] = value.to(dtype=_PAST_SUM_DTYPE)
        for key, value in out.items():
            if key not in sums and torch.is_tensor(value):
                sums[key] = value.to(dtype=_PAST_SUM_DTYPE)
        return cls(past=out, past_sum=sums)


def _past_tensor_for_name(
    state: ELLAState,
    name: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Return a historical tensor, caching device copies for repeated steps."""
    candidates = [name]
    if not name.startswith("module."):
        candidates.append(f"module.{name}")

    resolved_name: str | None = None
    tensor: torch.Tensor | None = None
    for candidate in candidates:
        if candidate in state.past_sum:
            resolved_name = candidate
            tensor = state.past_sum[candidate]
            break
        if candidate in state.past and torch.is_tensor(state.past[candidate]):
            resolved_name = candidate
            tensor = state.past[candidate]  # legacy/in-memory compatibility
            break

    if tensor is None:
        # New states store historical low-rank factors. Reconstruct the dense
        # historical delta only for this penalty evaluation; the state itself
        # does not retain a dense past_sum for factor-based entries.
        resolved = _past_factors_for_name(state, name)
        if resolved is None:
            return None
        resolved_name, factors = resolved
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        factor_cache_key = (resolved_name, str(target_device))
        cached_factors = state._past_factor_cache.get(factor_cache_key)
        if cached_factors is None:
            cached_factors = [
                (
                    b.detach().to(device=target_device, dtype=_PAST_SUM_DTYPE),
                    a.detach().to(device=target_device, dtype=_PAST_SUM_DTYPE),
                )
                for b, a in factors
            ]
            state._past_factor_cache[factor_cache_key] = cached_factors
        factors = cached_factors
        historical: torch.Tensor | None = None
        for b, a in factors:
            product = b @ a
            historical = product if historical is None else historical + product
        return historical.detach() if historical is not None else None
    if device is None:
        return tensor

    target_device = torch.device(device)
    # Keep the persistent GPU cache compact regardless of the model's
    # compute dtype (the training model is commonly bfloat16).
    target_dtype = _PAST_SUM_DTYPE
    if tensor.device == target_device and tensor.dtype == target_dtype:
        return tensor

    assert resolved_name is not None
    cache_key = (resolved_name, str(target_device), str(target_dtype))
    cached = state._past_tensor_cache.get(cache_key)
    if cached is None:
        cached = tensor.to(device=target_device, dtype=target_dtype)
        state._past_tensor_cache[cache_key] = cached
    return cached


def _past_key_for_update(state: ELLAState, name: str) -> str:
    if name in state.past or name in state.past_sum:
        return name
    if not name.startswith("module."):
        prefixed = f"module.{name}"
        if prefixed in state.past or prefixed in state.past_sum:
            return prefixed
    return name


def _has_past_for_name(state: ELLAState, name: str) -> bool:
    candidates = [name]
    if not name.startswith("module."):
        candidates.append(f"module.{name}")
    return any(
        candidate in state.past_sum
        or candidate in state.past
        for candidate in candidates
    )


def _past_factors_for_name(
    state: ELLAState,
    name: str,
) -> tuple[str, list[tuple[torch.Tensor, torch.Tensor]]] | None:
    candidates = [name]
    if name.startswith("module."):
        candidates.append(name.removeprefix("module."))
    else:
        candidates.append(f"module.{name}")

    for key in candidates:
        value = state.past.get(key)
        if not isinstance(value, (list, tuple)) or not value:
            continue
        factors = []
        for factor in value:
            if not isinstance(factor, (list, tuple)) or len(factor) != 2:
                raise TypeError(f"ELLA state key '{key}' has invalid factor entry.")
            b, a = factor
            if not torch.is_tensor(b) or not torch.is_tensor(a):
                raise TypeError(f"ELLA state key '{key}' has non-tensor factors.")
            factors.append((b, a))
        return key, factors
    return None


def _build_subspace_cache(
    state: ELLAState,
    name: str,
    device: torch.device,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor] | None:
    """Return historical factors and a Gram-matrix inverse square root."""
    resolved = _past_factors_for_name(state, name)
    if resolved is None:
        return None

    state_key, factors = resolved
    cache_key = (state_key, str(device))
    cached = state._subspace_cache.get(cache_key)
    if cached is not None:
        return cached

    # The Gram matrix is only task_count x task_count. Building it in FP64 on
    # CPU makes the rank decision stable without materializing dense past deltas.
    raw_cpu_factors = [
        (b.detach().cpu().double(), a.detach().cpu().double())
        for b, a in factors
    ]
    cpu_factors = []
    for b, a in raw_cpu_factors:
        norm_squared = torch.sum((b.t() @ b) * (a @ a.t()))
        if not torch.isfinite(norm_squared) or norm_squared <= 0:
            cpu_factors.append((torch.zeros_like(b), a))
        else:
            cpu_factors.append((b / norm_squared.sqrt(), a))
    task_count = len(cpu_factors)
    gram = torch.empty((task_count, task_count), dtype=torch.float64)
    for i, (b_i, a_i) in enumerate(cpu_factors):
        for j in range(i, task_count):
            b_j, a_j = cpu_factors[j]
            inner = torch.sum((b_i.t() @ b_j) * (a_i @ a_j.t()))
            gram[i, j] = inner
            gram[j, i] = inner

    gram = 0.5 * (gram + gram.t())
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    largest = eigenvalues[-1]
    if not torch.isfinite(largest) or largest <= 0:
        whitener = torch.empty((task_count, 0), dtype=torch.float32, device=device)
    else:
        keep = eigenvalues > largest * _SUBSPACE_EIGENVALUE_RTOL
        whitener = (
            eigenvectors[:, keep] * eigenvalues[keep].rsqrt().unsqueeze(0)
        ).to(device=device, dtype=torch.float32)

    device_factors = [
        (
            b.to(device=device, dtype=torch.float32),
            a.to(device=device, dtype=torch.float32),
        )
        for b, a in cpu_factors
    ]
    result = (device_factors, whitener)
    state._subspace_cache[cache_key] = result
    return result


def _subspace_projection_penalty(
    delta: torch.Tensor,
    state: ELLAState,
    name: str,
    past_sum: torch.Tensor,
) -> torch.Tensor:
    subspace_data = _build_subspace_cache(state, name, delta.device)
    with torch.autocast(device_type=delta.device.type, enabled=False):
        delta_f = delta.float()
        if subspace_data is None:
            # Legacy states only store the accumulated delta, so the complete
            # task subspace cannot be recovered. Preserve its one known direction.
            past_f = past_sum.to(device=delta.device, dtype=torch.float32)
            denominator = torch.sum(past_f.square()).clamp_min(1e-12)
            return torch.sum(delta_f * past_f).square() / denominator

        factors, whitener = subspace_data
        if whitener.shape[1] == 0:
            return delta_f.sum() * 0.0
        overlaps = torch.stack([
            torch.sum((b.t() @ delta_f) * a)
            for b, a in factors
        ])
        coordinates = whitener.t() @ overlaps
        return torch.sum(coordinates.square())


def compute_ella_penalty(
    deltas: Mapping[str, torch.Tensor],
    state: ELLAState,
    normalize_by_num_layers: bool = False,
    loss_type: str = "ella",
    subtract_past_tensor: bool = False,
) -> torch.Tensor:
    """Compute ELLA regularization term: sum || DeltaW_t ⊙ W_past ||_F^2."""
    if not deltas:
        return torch.tensor(0.0)

    total: torch.Tensor | None = None
    count = 0

    for name, delta in deltas.items():
        past_sum = _past_tensor_for_name(
            state,
            name,
            device=delta.device,
            dtype=delta.dtype,
        )
        if past_sum is None:
            continue
        if subtract_past_tensor:
            delta = delta - past_sum
        if loss_type == "l4_subspace":
            term = _subspace_projection_penalty(delta, state, name, past_sum)
            total = term if total is None else total + term
            count += 1
            continue
        past_tensor = past_sum
        if delta.dtype != _PAST_SUM_DTYPE:
            delta = delta.to(dtype=_PAST_SUM_DTYPE)
        if loss_type == "ella":
            term = torch.sum((delta * past_tensor) ** 2, dtype=torch.float32)
        elif loss_type in ("l3"):
            term = torch.sum(torch.sum(delta * past_tensor, dim=1) ** 2)
        elif loss_type in ("l3_normalized"):
            term = torch.sum(
                torch.sum(delta * past_tensor, dim=1) ** 2
                / (torch.sum(past_tensor * past_tensor, dim=1) + 1e-12)
            )
        elif loss_type in ("l4", "l4_proj"):
            term = torch.sum(delta * past_tensor) ** 2
        elif loss_type in ("l4_abs"):
            term = torch.abs(torch.sum(delta * past_tensor))
        elif loss_type in ("l4_normalized"):
            term = (torch.sum(delta * past_tensor)) ** 2 / (torch.sum(past_tensor * past_tensor) + 1e-12)
        elif loss_type == "l4_per_task":
            factors = state.past[name] if name in state.past else state.past[f"module.{name}"]
            task_terms = []
            for b, a in factors:
                b = b.to(device=delta.device, dtype=delta.dtype)
                a = a.to(device=delta.device, dtype=delta.dtype)
                past_task = b @ a
                task_terms.append(torch.sum(delta * past_task) ** 2)
            term = torch.stack(task_terms).mean()
        elif loss_type == "l4_abs_per_task":
            factors = state.past[name] if name in state.past else state.past[f"module.{name}"]
            task_terms = []
            for b, a in factors:
                b = b.to(device=delta.device, dtype=delta.dtype)
                a = a.to(device=delta.device, dtype=delta.dtype)
                past_task = b @ a
                task_terms.append(torch.abs(torch.sum(delta * past_task)))
            term = torch.stack(task_terms).mean()
        elif loss_type == "l4_normalized_per_task":
            factors = state.past[name] if name in state.past else state.past[f"module.{name}"]
            task_terms = []
            for b, a in factors:
                b = b.to(device=delta.device, dtype=delta.dtype)
                a = a.to(device=delta.device, dtype=delta.dtype)
                past_task = b @ a
                task_terms.append((torch.sum(delta * past_task)) ** 2 / (torch.sum(past_task * past_task) + 1e-12))
            term = torch.stack(task_terms).mean()
        total = term if total is None else total + term
        count += 1

    if total is None:
        return next(iter(deltas.values())).new_tensor(0.0)

    if normalize_by_num_layers and count > 0:
        total = total / count

    return total


def update_past_weights(
    state: ELLAState,
    deltas: Mapping[str, object],
) -> None:
    """Append the current task's LoRA factors to the past state.

    Values may be ``(B, A)`` or ``{"B": B, "A": A}``. Tensor values are
    retained as a compatibility path for callers that still provide deltas.
    """
    state._subspace_cache.clear()
    state._past_tensor_cache.clear()
    state._past_factor_cache.clear()
    for name, value in deltas.items():
        key = _past_key_for_update(state, name)
        if torch.is_tensor(value):
            d = value.detach().cpu()
            d_sum = d.to(dtype=_PAST_SUM_DTYPE)
            if key in state.past and torch.is_tensor(state.past[key]):
                state.past[key] = state.past[key] + d
            elif key not in state.past:
                state.past[key] = d.clone()
            else:
                raise TypeError(f"Cannot mix tensor and factor-list state for '{key}'.")
            previous = state.past_sum.get(key)
            state.past_sum[key] = d_sum if previous is None else previous.to(dtype=_PAST_SUM_DTYPE) + d_sum
            continue
        if isinstance(value, Mapping):
            b, a = value.get("B"), value.get("A")
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            b, a = value
        else:
            b, a = None, None
        if not torch.is_tensor(b) or not torch.is_tensor(a):
            raise TypeError(f"Expected (B, A) factors for '{name}'.")
        factors = state.past.setdefault(key, [])
        if torch.is_tensor(factors):
            raise TypeError(f"Cannot mix tensor and factor-list state for '{key}'.")
        b_cpu, a_cpu = b.detach().cpu().clone(), a.detach().cpu().clone()
        factors.append((b_cpu, a_cpu))  # type: ignore[union-attr]
        # Factor-based history remains compact. A dense historical delta is
        # reconstructed only when the penalty for this layer is evaluated.


def compute_ella_penalty_from_model(
    model: torch.nn.Module,
    state: ELLAState,
    normalize_by_num_layers: bool = True,
    loss_type: str = "ella",
    delta_mode: str = "layerwise",
    subtract_past_tensor: bool = False,
) -> torch.Tensor:
    """Compute the ELLA penalty using either layerwise or all-at-once deltas.

    ``layerwise`` avoids materializing the complete delta mapping before
    computing the penalty. ``all`` preserves the original behavior and is
    useful for comparisons or callers that prefer the simpler execution path.
    """
    if delta_mode == "all":
        deltas = collect_lora_deltas(model)
        return compute_ella_penalty(
            deltas=deltas,
            state=state,
            normalize_by_num_layers=normalize_by_num_layers,
            loss_type=loss_type,
            subtract_past_tensor=subtract_past_tensor,
        )
    if delta_mode != "layerwise":
        raise ValueError("delta_mode must be either 'layerwise' or 'all'.")

    total: torch.Tensor | None = None
    count = 0
    for name, delta in iter_lora_deltas(model):
        # Count only layers that actually have a historical ELLA direction,
        # matching compute_ella_penalty's normalization semantics.
        if _has_past_for_name(state, name):
            count += 1
        term = compute_ella_penalty(
            deltas={name: delta},
            state=state,
            normalize_by_num_layers=False,
            loss_type=loss_type,
            subtract_past_tensor=subtract_past_tensor,
        )
        total = term if total is None else total + term

    if total is None:
        parameter = next(model.parameters(), None)
        return parameter.new_tensor(0.0) if parameter is not None else torch.tensor(0.0)

    if normalize_by_num_layers and count > 0:
        total = total / count
    return total


def project_lora_deltas_orthogonal(
    state: ELLAState,
    model: torch.nn.Module,
) -> Dict[str, Dict[str, float]]:
    """Project active PEFT LoRA deltas away from accumulated past deltas.

    Each projected matrix is compressed back to the configured adapter rank
    with a truncated SVD, corrected for the orthogonality constraint, and then
    written into the active adapter's B/A weights.
    """
    stats: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None:
                continue

            adapter = _active_adapter_name(module)
            if adapter not in lora_a or adapter not in lora_b:
                continue

            adapter_a = lora_a[adapter].weight
            adapter_b = lora_b[adapter].weight
            scaling = module.scaling.get(adapter, 1.0) if hasattr(module, "scaling") else 1.0
            scaling_value = float(scaling)
            if scaling_value == 0.0:
                continue

            module_name = _normalize_module_name(name)
            past = _past_tensor_for_name(state, module_name)
            if past is None:
                continue

            delta = (adapter_b @ adapter_a) * scaling_value
            fan_in_fan_out = bool(getattr(module, "fan_in_fan_out", False))
            if fan_in_fan_out:
                delta = delta.t()
            past = past.to(device=delta.device, dtype=delta.dtype)

            denominator = torch.sum(past * past)
            if denominator.item() <= torch.finfo(delta.dtype).eps:
                continue
            inner_before = torch.sum(delta * past)
            projected = delta - (inner_before / denominator) * past

            raw_stored = projected / scaling_value
            if fan_in_fan_out:
                raw_stored = raw_stored.t()
            svd_input = (
                raw_stored.float()
                if raw_stored.dtype in (torch.float16, torch.bfloat16)
                else raw_stored
            )
            u, singular_values, vh = torch.linalg.svd(svd_input, full_matrices=False)
            rank = min(adapter_a.shape[0], u.shape[1], vh.shape[0])
            factor_scale = singular_values[:rank].clamp_min(0).sqrt()

            # Truncation can reintroduce overlap with W_past. Correct one
            # factor row after the SVD while preserving the adapter rank.
            factor_b = (u[:, :rank] * factor_scale.unsqueeze(0)).float()
            factor_a = (factor_scale.unsqueeze(1) * vh[:rank, :]).float()
            past_stored = past.t() if fan_in_fan_out else past
            past_stored = past_stored.float()
            raw_inner = torch.sum((factor_b @ factor_a) * past_stored)
            q = factor_b.t() @ past_stored
            q_norms = torch.sum(q * q, dim=1)
            correction_index = int(torch.argmax(q_norms).item())
            q_norm = q_norms[correction_index]
            if q_norm.item() > torch.finfo(torch.float32).eps:
                factor_a[correction_index] -= (raw_inner / q_norm) * q[correction_index]

            new_b = torch.zeros_like(adapter_b)
            new_a = torch.zeros_like(adapter_a)
            new_b[:, :rank] = factor_b.to(new_b.dtype)
            new_a[:rank, :] = factor_a.to(new_a.dtype)
            adapter_b.copy_(new_b)
            adapter_a.copy_(new_a)

            reconstructed = (adapter_b @ adapter_a) * scaling_value
            if fan_in_fan_out:
                reconstructed = reconstructed.t()
            stats[module_name] = {
                "inner_product_before": float(inner_before.float().cpu().item()),
                "inner_product_projected": float(torch.sum(projected * past).float().cpu().item()),
                "inner_product_after_svd": float(torch.sum(reconstructed * past).float().cpu().item()),
                "projection_coefficient": float((inner_before / denominator).float().cpu().item()),
            }
    return stats


def update_past_weights_from_model(state: ELLAState, model: torch.nn.Module) -> None:
    factors = {}
    for name, module in model.named_modules():
        if hasattr(module, "A") and hasattr(module, "B") and hasattr(module, "scaling"):
            factors[name.removeprefix("module.")] = (module.B * module.scaling, module.A)
            continue
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        adapter = getattr(module, "active_adapter", None)
        if adapter is None:
            adapter = getattr(module, "active_adapters", "default")
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0]
        if adapter in lora_a and adapter in lora_b:
            b, a = lora_b[adapter].weight, lora_a[adapter].weight
            scaling = module.scaling.get(adapter, 1.0) if hasattr(module, "scaling") else 1.0
            if getattr(module, "fan_in_fan_out", False):
                factors[name.removeprefix("module.")] = (a.t(), b.t() * scaling)
            else:
                factors[name.removeprefix("module.")] = (b * scaling, a)
    update_past_weights(state=state, deltas=factors)
