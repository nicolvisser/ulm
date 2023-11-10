import sentencepiece as spm
import torch
from ulm.module import LitGPT, GPTConfig
import json

sp = spm.SentencePieceProcessor(model_file="character_level.model")

prompt = "mister"

ids = sp.EncodeAsIds(prompt)

ids = torch.tensor(ids, dtype=torch.long)

ids = ids.unsqueeze(0).cuda()


with open("config.json", "r") as f:
    config = GPTConfig(**json.load(f))

model = LitGPT.load_from_checkpoint(
    "/home/nicolvisser/workspace/ulm/lightning_logs/text_test/checkpoints/epoch=9-step=3120.ckpt",
    config=config,
)


result = model.generate(ids, 100, temperature=0.5, top_k=None)

result = result.tolist()[0]


print(sp.DecodeIds(result))
