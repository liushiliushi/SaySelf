MODEL="output/qwen_lr7e-5_bs128"
train_file=../datasets/stage_2/hotpotqa.json
NUM_GPUS=1
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=8 # max 2 for 2 GPUs
LEARNING_RATE=1e-5
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
OUTPUT_DIR=qwen_lr${LEARNING_RATE}_bs${TOTAL_BATCH_SIZE}
echo "Training model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 29600 \
    --deepspeed_config_file ../ds_configs/stage2_no_offloading_accelerate.conf \
    ./rlhf_train.py \
    --peft_model_name_or_path $MODEL \
    --use_flash_attn \
    --train_file ${train_file} \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --train_micro_batch_size_per_gpu $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --num_train_epochs 2 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --num_train_examples 2000 \
