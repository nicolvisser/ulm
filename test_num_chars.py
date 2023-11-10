from pathlib import Path
import click

path = Path(click.prompt("Path to text file: ", type=str))

text = path.read_text("utf-8")

# remove first two lines
text = text.split("\n")[2:]
text = "\n".join(text)

print(len(list(set(text))))
