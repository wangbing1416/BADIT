export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

lr=0.0002
lora_config=$2

MODEL_DIR=../huggingface/qwen3-8b

pretrained_model=$MODEL_DIR
tokenizer_path=$MODEL_DIR
dataset_dir=./data/SuperNI
validation_file=./data/superNI

per_device_train_batch_size=1
per_device_eval_batch_size=8
gradient_accumulation_steps=1
max_seq_length=1024
seed=$1
lora_name=$3
task_name=$4
output_dir=./output
exp_name=qwen3-8b-${lora_name}-baseline

deepspeed_config_file=ds_zero2_no_offload.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_LAUNCH_BLOCKING=1 \
deepspeed --num_nodes 1 --num_gpus 8 run_loramoe_baseline.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --dataloader_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --bf16 \
    --seed ${seed} \
    --num_train_epochs 10 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir ./tensorboard_log \
    --save_strategy steps \
    --save_total_limit 5 \
    --eval_strategy no \
    --eval_steps 5000 \
    --save_steps 5000 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --max_target_length 50 \
    --output_dir ${output_dir}/${exp_name} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_config_file ${lora_config} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --attn_implementation flash_attention_2 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --task_name ${task_name}
