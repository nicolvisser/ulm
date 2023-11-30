from lightning.pytorch.callbacks import Callback

import torch
import torchaudio
import wandb

from ulm.module import LitGPT
import sentencepiece as spm


class LogSampleContinuationCallback(Callback):
    def __init__(
        self,
        prompt_audio_path,
        n_units,
        dp_lambda,
        tokenizer_path,
    ):
        self.n_units = n_units
        self.dp_lambda = dp_lambda

        self.gslm = torch.hub.load(
            "nicolvisser/gslm-hubert-hifigan:master",
            "gslm",
            n_units=n_units,
            dp_lambda=dp_lambda,
            trust_repo=True,
        )
        self.gslm.to("cpu")

        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

        prompt_wav, sr = torchaudio.load(prompt_audio_path)
        prompt_wav = prompt_wav.to("cpu")
        prompt_unicode = self.gslm.encode_unicode(prompt_wav, sr, dedupe=True)
        self.prompt_ids = torch.tensor(self.sp.EncodeAsIds(prompt_unicode)).unsqueeze(0)

    def on_validation_epoch_end(self, trainer, pl_module: LitGPT):
        continuation_ids = pl_module.generate_top_p(
            self.prompt_ids.to(pl_module.device),
            300,
            top_p=0.8,
        )
        continuation_ids.to("cpu")

        continuation_unicode = self.sp.DecodeIds(continuation_ids.tolist()[0])

        # remove all the chars not in the range 0x4E00 to 0x4E00 + n_units
        stripped = []
        for c in continuation_unicode:
            if ord(c) in range(0x4E00, 0x4E00 + self.n_units):
                stripped.append(c)
            else:
                break
        wav_, sr_ = self.gslm.decode_unicode(stripped, deduped=True)

        trainer.logger.experiment.log(
            {
                "val/audio": wandb.Audio(
                    wav_.squeeze().numpy(),
                    sample_rate=sr_,
                    caption="Sample Continuation",
                ),
            }
        )
