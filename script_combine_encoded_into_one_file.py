from pathlib import Path
from tqdm import tqdm
import click


input_dir = click.prompt("Enter input directory", type=str)
output_path = click.prompt("Enter output path (.txt)", type=str)

train_files = sorted(list(Path(input_dir).rglob("*.txt")))
for i, train_file in enumerate(tqdm(train_files)):
    text = train_file.read_text("utf-8")
    # append text to output file
    with open(output_path, "a") as f:
        if i == 0:
            f.write(text)
        else:
            f.write("\n")
            f.write(text)
