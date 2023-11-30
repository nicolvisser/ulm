python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/dev.pt \
    --version_name ulm-500-16-bpe5k \
    --batch_size 48 \
    --learning_rate 0.00005 \
    --num_workers 64 \
    --max_steps 100000 \
    --train_num_samples 1000 \
    --val_num_samples 1000