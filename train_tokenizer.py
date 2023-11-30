import click
import sentencepiece as spm
import torch
from pathlib import Path

train_text_path = Path(
    click.prompt(
        "Path to train text file: ",
        type=click.Path(exists=True, dir_okay=False),
    )
)
val_text_path = Path(
    click.prompt(
        "Path to val text file: ",
        type=click.Path(exists=True, dir_okay=False),
    )
)
output_dir = Path(
    click.prompt(
        "Output directory: ",
        type=click.Path(exists=True, file_okay=False),
    )
)

with open(train_text_path, "r") as f:
    train_text = f.read()

with open(val_text_path, "r") as f:
    val_text = f.read()

text = train_text + val_text

init_vocab = sorted(list(set(text)))

init_vocab_size = len(init_vocab)

model_prefix = output_dir / "tokenizer"

print("Training tokenizer...")

spm.SentencePieceTrainer.train(
    input=train_text_path,
    model_prefix=model_prefix,
    model_type="char",
    character_coverage=1.0,
    input_sentence_size=10000000000,  # basically use all sentences
    shuffle_input_sentence=False,
)

print("Done!")

sp = spm.SentencePieceProcessor(model_file=str(model_prefix.with_suffix(".model")))

print(f"Vocab size: {sp.vocab_size()}")

print("Encoding train and val text...")

train = torch.tensor(sp.EncodeAsIds(train_text), dtype=torch.long)
val = torch.tensor(sp.EncodeAsIds(val_text), dtype=torch.long)

print("Done!")

torch.save(train, output_dir / "train.pt")
torch.save(val, output_dir / "dev.pt")

print(f"Saved train.pt and val.pt to {output_dir}")
