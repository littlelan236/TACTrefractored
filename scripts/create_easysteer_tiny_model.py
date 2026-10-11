"""Make a labelled random fixture for native API tests; no research evaluation."""

from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast

root = Path(__file__).resolve().parents[1]
directory = root / "assets/test-models/easysteer-tiny-qwen2"
if not directory.exists():
    torch.manual_seed(19)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(config).save_pretrained(directory)
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "hello": 1, "</think>": 2, **{f"w{i}": i for i in range(3, 256)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=["</think>"]
    )
    fast.save_pretrained(directory)
print(directory)
