from pathlib import Path
from tqdm import tqdm
import click


input_dir = Path(click.prompt("Enter input directory", type=str))

train_files = sorted(list(Path(input_dir).rglob("train*/**/*.txt")))
dev_files = sorted(list(Path(input_dir).rglob("dev*/**/*.txt")))

train_txt_path = input_dir / "train.txt"
dev_txt_path = input_dir / "dev.txt"

for files, txt_path in zip([train_files, dev_files], [train_txt_path, dev_txt_path]):
    for i, file in enumerate(tqdm(files)):
        text = file.read_text("utf-8")
        # append text to output file
        with open(txt_path, "a") as f:
            if i == 0:
                f.write(text)
            else:
                f.write("\n")
                f.write(text)
