python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/dev.pt \
    --version_name ulm-500-16-bpe5k \
    --batch_size 50 \
    --learning_rate "1e-5" \
    --num_workers 64 \
    --max_steps 100000 \
    --train_num_samples 50000 \
    --val_num_samples 2500 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/dev.pt \
    --version_name ulm-500-16-bpe5k \
    --batch_size 50 \
    --learning_rate "5e-4" \
    --num_workers 64 \
    --max_steps 100000 \
    --train_num_samples 50000 \
    --val_num_samples 2500 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

# compare after 3000 steps