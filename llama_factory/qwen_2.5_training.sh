export NPROC_PER_NODE=4
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR="localhost"
export MASTER_PORT=12356

export DS_CONFIG_PATH="examples/deepspeed/ds_z3_config.json"
export MODEL_PATH="/fdudata/share_models/Qwen2.5-7B-Instruct"
export OUTPUT_PATH="/fdudata/qyliu/models/wuguan_qwen2.5_7b_ver1/"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
  --deepspeed $DS_CONFIG_PATH \
  --stage sft \
  --do_train \
  --use_fast_tokenizer \
  --flash_attn auto \
  --model_name_or_path $MODEL_PATH \
  --dataset wuguan_ver1 \
  --template qwen \
  --finetuning_type full \
  --overwrite_cache \
  --overwrite_output_dir \
  --warmup_ratio 0.1 \
  --weight_decay 0.05 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --ddp_timeout 90000 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --logging_steps 100 \
  --cutoff_len 3072 \
  --save_steps 1000 \
  --plot_loss \
  --num_train_epochs 2 \
  --bf16