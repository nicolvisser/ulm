python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-0/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-0/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-0/dev.pt \
    --version_name ulm-500-0 \
    --n_units 500 \
    --dp_lambda 0 \
    --tokenizer_path /home/nicolvisser/workspace/ulm/data/500-0-bpe5k/tokenizer.model \
    --num_workers 64 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-16/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-16/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-16/dev.pt \
    --version_name ulm-500-16 \
    --n_units 500 \
    --dp_lambda 16 \
    --tokenizer_path /home/nicolvisser/workspace/ulm/data/500-16/tokenizer.model \
    --num_workers 64 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-0-bpe5k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-0-bpe5k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-0-bpe5k/dev.pt \
    --version_name ulm-500-0-bpe5k \
    --n_units 500 \
    --dp_lambda 0 \
    --tokenizer_path /home/nicolvisser/workspace/ulm/data/500-0-bpe5k/tokenizer.model \
    --num_workers 64 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-0-bpe10k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-0-bpe10k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-0-bpe10k/dev.pt \
    --version_name ulm-500-0-bpe10k \
    --n_units 500 \
    --dp_lambda 0 \
    --tokenizer_path /home/nicolvisser/workspace/ulm/data/500-0-bpe10k/tokenizer.model \
    --num_workers 64 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/dev.pt \
    --version_name ulm-500-16-bpe5k \
    --n_units 500 \
    --dp_lambda 16 \
    --tokenizer_path /home/nicolvisser/workspace/ulm/data/500-16-bpe5k/tokenizer.model \
    --num_workers 64 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1

python train_ulm.py \
    --config_path /home/nicolvisser/workspace/ulm/data/500-16-bpe10k/config.json \
    --train_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe10k/train.pt \
    --val_ids_path /home/nicolvisser/workspace/ulm/data/500-16-bpe10k/dev.pt \
    --version_name ulm-500-16-bpe10k \
    --n_units 500 \
    --dp_lambda 16 \
    --tokenizer_path /home/nicolvisser/workspace/ulm/data/500-16-bpe10k/tokenizer.model \
    --num_workers 64 \
    --val_check_interval 1.0 \
    --log_every_n_steps 1
