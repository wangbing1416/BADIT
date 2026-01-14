export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

lr=0.0002
lora_config="config/lora.config"

MODEL_DIR=../huggingface/llama3.2-3b-instruct

pretrained_model=$MODEL_DIR
tokenizer_path=$MODEL_DIR
dataset_dir=./data/SuperNI
validation_file=./data/superNI

per_device_eval_batch_size=8
gradient_accumulation_steps=1
max_seq_length=1024
seed=${1:-1}
output_dir=./analysis
exp_name=llama3-3b-analysis-1025

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CUDA_LAUNCH_BLOCKING=1 \

task_num_list=(
    "task1590_diplomacy_text_generation:4"
    "task875_emotion_classification:32"
    "task511_reddit_tifu_long_text_summarization:4"
    "task1572_samsum_summary:8"
    "task591_sciq_answer_generation:32"
    "task002_quoref_answer_generation:4"
    "task639_multi_woz_user_utterance_generation:32"
    "task748_glucose_reverse_cause_event_detection:24"
    "task1290_xsum_summarization:4"
    "task1510_evalution_relation_extraction:32"
    "task363_sst2_polarity_classification:32"
    "task181_outcome_extraction:16"
    "task1687_sentiment140_classification:32"
    "task1729_personachat_generate_next:8"
    "task073_commonsenseqa_answer_generation:32"
)

# Loop through each task
for item in "${task_num_list[@]}"; do
    task="${item%%:*}"
    per_device_train_batch_size="${item##*:}"
    echo "Processing task: $task with batch size: $per_device_train_batch_size"

    torchrun --nnodes 1 --nproc_per_node 8 --node_rank 0 --master_port 29502 analysis/analysis_grad.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --task_name ${task} \
    --dataloader_num_workers 4 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --bf16 \
    --seed ${seed} \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length ${max_seq_length} \
    --max_target_length 50 \
    --output_dir ${output_dir}/${exp_name} \
    --ddp_timeout 30000 \
    --lora_config_file ${lora_config} \
    --torch_dtype float16 \
    --ddp_find_unused_parameters False \
    --flash_attn \
    --attn_implementation eager \
    --overwrite_output_dir \
    --eval_strategy no \
    --save_strategy no \
    --logging_strategy no \
    --report_to none \
    --no_cuda false
done
