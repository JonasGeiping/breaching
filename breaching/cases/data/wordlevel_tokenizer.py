"""Small helper script to generate a word-level tokenizer.
In a FL setting this tokenizer might be trained on a public list of words and shared with all users,
as no central party has access to the dataset.

We'll use AG-News as a substitute public dataseet for up to 62k tokens, and wikitext up to 215k tokens.
(Not that I think a 215k token word-level tokenizer would be a great idea anyways)
"""


"""This is code from https://huggingface.co/robot-test/dummy-tokenizer-wordlevel."""
import os

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from datasets import load_dataset


def generate_word_level_tokenizer(vocab_size=50_000, cache_dir="~/data"):
    if vocab_size < 62_000:
        dataset = load_dataset("ag_news", cache_dir=cache_dir, split="train")
    elif vocab_size < 215000:
        dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=cache_dir, split="train")
    else:
        raise ValueError("Not enough data to create a word-level tokenizer of this size.")
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )

    def batch_iterator(batch_size=1024):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))
    os.makedirs(os.path.join(cache_dir, "cache"), exist_ok=True)
    path = os.path.join(cache_dir, "cache", f"word-tokenizer_{vocab_size}.json")
    tokenizer.save(path)


if __name__ == "__main__":
    generate_word_level_tokenizer()
