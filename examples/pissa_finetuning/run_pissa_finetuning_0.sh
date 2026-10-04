cd /home/fit/lishbo/WORK/zzl/repo/peft
BASE_MODEL="/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Llama-2-7b-hf"
# BASE_MODEL="/home/fit/lishbo/WORK/data/zhengzhilong/anticf/pissa_residual_model/Meta-Llama-3-8B"
FT_MODEL="metamath-pissa-llama-2-7b_ella_lambda50_seed233"
OUTPUT_PATH="output/$FT_MODEL"
DATA_PATH="fxmeng/pissa-dataset"


deepspeed --master_port=16972 --include=localhost:0,1 examples/pissa_finetuning/my_pissa_finetuning.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT_PATH \
    --pissa_mode True \
    --data_path $DATA_PATH \
    --dataset_split train \
    --sub_task metamath:100000 \
    --dataset_field instruction output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 3 \
    --save_only_model True \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --tf32 True \
    --seed 233 \
    --report_to "tensorboard" \
    --ella_lambda 50 \
    --ella_loss_type "ella" \
    --ella_delta_mode "layerwise" \
    --bf16 True \


# cd /home/fit/lishbo/WORK/zzl/repo/peft/output
# OUTPUT_PATH="../result"
# CACHE_PATH="../result/$FT_MODEL"

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=$FT_MODEL,use_fast_tokenizer=True" \
#     --tasks triviaqa,nq_open,webqs \
#     --batch_size 128 \
#     --output_path $OUTPUT_PATH \

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=$FT_MODEL,use_fast_tokenizer=False" \
#     --tasks triviaqa,nq_open,webqs \
#     --batch_size 128 \
#     --output_path $OUTPUT_PATH \