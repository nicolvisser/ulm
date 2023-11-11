import click
import sentencepiece as spm
import torch
from pathlib import Path

train_text_path = click.prompt("Path to train text file: ", type=str)
val_text_path = click.prompt("Path to val text file: ", type=str)

with open(train_text_path, "r") as f:
    train_text = f.read()

with open(val_text_path, "r") as f:
    val_text = f.read()

text = train_text + val_text

init_vocab = sorted(list(set(text)))

init_vocab_size = len(init_vocab)

spm.SentencePieceTrainer.train(
    input=train_text_path,
    model_prefix="bpe",
    model_type="bpe",
    vocab_size=30000,
    character_coverage=1.0,
    input_sentence_size=10000000000,
    shuffle_input_sentence=False,
)

sp = spm.SentencePieceProcessor(model_file="bpe.model")

train = torch.tensor(sp.EncodeAsIds(train_text), dtype=torch.long)
val = torch.tensor(sp.EncodeAsIds(val_text), dtype=torch.long)

print(f"Vocab size: {sp.vocab_size()}")

output_dir = click.prompt("Output directory: ", type=str)
output_dir = Path(output_dir)

torch.save(train, output_dir / "train.pt")
torch.save(val, output_dir / "val.pt")
