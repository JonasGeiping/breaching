"""Small helper script to generate a word-level tokenizer.
In a FL setting this tokenizer might be trained on a public list of words and shared with all users,
as no central party has access to the dataset.

We'll use AG-News as a substitute public dataseet.
"""


"""This is code from https://huggingface.co/robot-test/dummy-tokenizer-wordlevel."""

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer

from datasets import load_dataset


def generate_word_level_tokenizer():
    dataset = load_dataset("ag_news", cache_dir="~/data", split="train")

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    trainer = WordLevelTrainer(vocab_size=50_000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))

    tokenizer.save("word-tokenizer.json")


if __name__ == "__main__":
    generate_word_level_tokenizer()
