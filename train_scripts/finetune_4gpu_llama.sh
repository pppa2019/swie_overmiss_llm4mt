export CUDA_VISIBLE_DEVICES=0,1,2,3
premodel=llama-7b-hf
code_path=run_clm_llm_llama.py
suffix=
train_data= # train dataset here
LOG_FILE=${premodel}_$suffix.log
ckpt_dir=output/${premodel}_$suffix
deepspeed src/run_clm_llm_llama.py  \
    --model_name_or_path yahma/llama-7b-hf \
    --train_file ${train_data} \
    --streaming \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --ignore_data_skip True \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --keep_linebreaks False \
    --logging_steps 10 \
    --save_total_limit 10 \
    --overwrite_cache \
    --overwrite_output_dir \
    --save_steps 200 \
    --adam_beta2 0.95 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "cosine" \
    --validation_split_percentage 0 \
    --deepspeed ./ds_config1.json \
    --do_train \
    --fp16 True \
    --seed 42 \
    --gradient_accumulation_steps 64  \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --block_size 512 \
    --only_optimize_layers "31" "30" "29" "28" "27" "26" "25" "24" "23" "22" "21" "20" "19" "18" "17" "16" "15" "14" "13" "12" "11" "10"  \
    --selected_layer   "14" "15" "16"  \
    --output_dir ${ckpt_dir} \
    --num_train_epochs 1.5 \
    --ins_pool_method max \
    --ins_fuse_method add \
    --ins_hidden_size 32 \
    --add_ins_transform True \
    --warm_up_alpha True \
    --cat_post_ins seg_sin \
    --origin_setting False \
    --qkv_only False \
2>&1 |tee log/${LOG_FILE}
