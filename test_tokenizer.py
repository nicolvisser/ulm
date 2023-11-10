import sentencepiece as spm

# Initialize SentencePiece processor with the trained model file
sp = spm.SentencePieceProcessor(model_file="character_level.model")

# Now you can use the model
encoded_pieces = sp.encode("丘丮上丮个下丬丗丛世且东丟丆万丄个下个下丬並七丣一七不丨丟丆丮丘", out_type=str)
print(encoded_pieces)

print(sp.EncodeAsIds("丘丮上丮个下丬丗丛世且东丟丆万丄个下个下丬並七丣一七不丨丟丆丮丘"))

print(sp.vocab_size())
