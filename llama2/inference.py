from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from model import ModelArgs, Transformer


class LLaMA:
    def __int__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoint_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints found in directory"
            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            check_point = torch.load(chk_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}s")
            prev_time = time.time()

        with open(Path(checkpoint_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args)
        if load_model:
            del check_point["rope.freqs"]
            model.load_state_dict(check_point, strict=True)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f}s")

        return LLaMA(model, tokenizer, model_args)


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    model = LLaMA.build(
        checkpoint_dir="checkpoints/llama2",
        tokenizer_path="data/llama2/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
    )
    print("All OK!")
