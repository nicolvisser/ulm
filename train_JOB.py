import os

for n_units in [500]:
    for dp_lambda in [0, 16]:
        for bpe_suffix in ["", "-bpe5k", "-bpe10k"]:
            command = []
            command.append("python")
            command.append("train_ulm.py")
            command.append("--config_path")
            command.append(f"/home/nicolvisser/workspace/ulm/data/500-{dp_lambda}{bpe_suffix}/config.json")
            command.append("--train_ids_path")
            command.append(f"/home/nicolvisser/workspace/ulm/data/500-{dp_lambda}{bpe_suffix}/train.pt")
            command.append("--val_ids_path")
            command.append(f"/home/nicolvisser/workspace/ulm/data/500-{dp_lambda}{bpe_suffix}/dev.pt")
            command.append("--version_name")
            command.append(f"ulm-500-{dp_lambda}{bpe_suffix}")
            command.append("--n_units")
            command.append(str(n_units))
            command.append("--dp_lambda")
            command.append(str(dp_lambda))
            command.append("--tokenizer_path")
            command.append(f"/home/nicolvisser/workspace/ulm/data/500-{dp_lambda}{bpe_suffix}/tokenizer.model")
            command.append("--num_workers")
            command.append("64")
            command.append("--val_check_interval")
            command.append("0.5")
            command.append("--log_every_n_steps")
            command.append("1")

            os.system(" ".join(command))