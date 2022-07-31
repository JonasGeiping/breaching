"""Glue code to include text data seamlessly (or like more or less ;>)"""
import torch
import os

from itertools import chain
import collections

# All language modules are import lazily
import logging

log = logging.getLogger(__name__)


def _build_and_split_dataset_text(cfg_data, split, user_idx=None, return_full_dataset=False):
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg_data.path = os.path.expanduser(cfg_data.path)
    from datasets import load_dataset, Dataset, disable_progress_bar

    disable_progress_bar()

    if user_idx is None:
        user_idx = torch.randint(0, cfg_data.default_clients, (1,)).item()
    else:
        if user_idx > cfg_data.default_clients:
            raise ValueError("This user index exceeds the maximal number of clients.")
    if split == "training":
        split = "train"  # huggingface notation

    cfg_data.path = os.path.expanduser(cfg_data.path)

    if cfg_data.name == "wikitext":  # random tokens shares the wikitext tokenizer
        raw_dataset = load_dataset("wikitext", "wikitext-103-v1", cache_dir=cfg_data.path, split=split)
        raw_dataset = _split_wikipedia_into_articles(raw_dataset, user_idx, return_full_dataset, min_length=25)
    elif cfg_data.name == "stackoverflow":
        raw_texts = load_stackoverflow(cfg_data.path, user_idx, return_full_dataset, split=split)
        raw_dataset = Dataset.from_dict(dict(text=raw_texts))
    elif cfg_data.name == "shakespeare":
        raw_texts = load_shakespeare(cfg_data.path, user_idx, return_full_dataset, split=split)
        raw_dataset = Dataset.from_dict(dict(text=raw_texts))
    elif cfg_data.name == "random-tokens":
        pass
    elif cfg_data.name == "cola":
        if return_full_dataset:
            raw_datapoint = load_dataset("glue", "cola", cache_dir=cfg_data.path)[split]
        else:
            raw_datapoint = load_dataset("glue", "cola", cache_dir=cfg_data.path)[split][user_idx]
        raw_dataset = Dataset.from_dict({k: [v] for k, v in raw_datapoint.items()})
    else:
        raise ValueError(f"Invalid text dataset {cfg_data.name} provided.")

    tokenizer = _get_tokenizer(cfg_data.tokenizer, cfg_data.vocab_size, cache_dir=cfg_data.path)
    tokenize, group_texts, collate_fn = _get_preprocessing(tokenizer, cfg_data)
    if cfg_data.name != "random-tokens":
        columns = raw_dataset.column_names
        if "label" in columns:
            columns.remove("label")
            raw_dataset = raw_dataset.rename_column("label", "labels")
        tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=columns, load_from_cache_file=False)
        tokenized_dataset = tokenized_dataset.map(group_texts, batched=True, load_from_cache_file=False)
    else:
        generator = torch.Generator()
        generator.manual_seed(user_idx + 233)
        random_tokens = torch.randint(0, cfg_data.vocab_size, (cfg_data.size, cfg_data.shape[0]), generator=generator)
        tokenized_dataset = Dataset.from_dict(dict(input_ids=random_tokens, labels=random_tokens))

    tokenized_dataset.set_format("torch")
    tokenized_dataset.tokenizer = tokenizer  # Stash here

    # Reduce train dataset according to cfg_data.size:
    if cfg_data.size < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.select(range(0, cfg_data.size))

    return tokenized_dataset, collate_fn


def _get_preprocessing(tokenizer, cfg_data):
    """
    preprocessing inspired by https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py#L432
    """
    from transformers import default_data_collator, DataCollatorForLanguageModeling

    def tokenize(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=cfg_data == "masked-lm")

    if "causal-lm" in cfg_data.task or "masked-lm" in cfg_data.task:
        block_size = cfg_data.shape[0]
        tokenizer.model_max_length = 1e10  # Only for batched pre-processing

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if "causal-lm" in cfg_data.task:
                result["labels"] = result["input_ids"].copy()
            return result

        if "causal-lm" in cfg_data.task:
            # This setting adds "labels" during "group_texts"
            collate_fn = default_data_collator
        else:
            # This collate_fn generates "labels" automatically after masking
            collate_fn = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=not cfg_data.disable_mlm, mlm_probability=cfg_data.mlm_probability
            )
    elif cfg_data.task == "classification":
        tokenizer.model_max_length = cfg_data.shape[0]

        def tokenize(examples):  # noqa F811
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)

        group_texts = None
        collate_fn = default_data_collator
    else:
        raise ValueError(f"Invalid task {cfg_data.task}")

    return tokenize, group_texts, collate_fn


def _get_tokenizer(tokenizer_name, vocab_size=None, cache_dir=None):
    """Load tokenizer."""
    from transformers import PreTrainedTokenizerFast, AutoTokenizer, CanineTokenizer

    if tokenizer_name == "word-level":
        path = os.path.join(cache_dir, "cache", f"word-tokenizer_{vocab_size}.json")
        if os.path.isfile(path):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        else:
            from .wordlevel_tokenizer import generate_word_level_tokenizer

            generate_word_level_tokenizer(vocab_size=vocab_size, cache_dir=cache_dir)
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=path, cache_dir=cache_dir)
    elif tokenizer_name == "character":
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c", cache_dir=cache_dir)
    elif tokenizer_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    elif tokenizer_name == "GPT-2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    elif tokenizer_name == "eleutherAI-GPT":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", cache_dir=cache_dir)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        except OSError as error_msg:
            raise ValueError(f"Invalid huggingface tokenizer {tokenizer_name} given: {error_msg}")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(f"Requested tokenizer with vocab_size {tokenizer.vocab_size} incompatible with given vocab.")
    return tokenizer


def _split_wikipedia_into_articles(dataset, user_idx=0, return_full_dataset=False, min_length=25):
    """Split along headlines, discard minor headers and tiny lines."""
    # annotate articles as separate users:
    # num_articles_estimate = len(
    #     [line for line in dataset["text"] if line.count("=") == 2 and len(line) < 100]
    # )  # this is good enough
    # print(f"Estimating {num_articles_estimate} articles in this wikipedia dump.")
    if not return_full_dataset:
        # Super-dirty selector:
        clean_line_ids = []
        article_counter = 0
        for idx, line in enumerate(dataset["text"]):
            if " = " in line and " ; " not in line:  # exclude table headers
                if line.count("=") == 2 and len(line) < 100:
                    article_counter += 1
                    # print(f"Checking article {article_counter}: {line} at idx {idx}")
            elif len(line) < min_length:
                pass
            else:
                if user_idx + 1 == article_counter:
                    clean_line_ids.append(idx)
            if article_counter > user_idx + 1:
                break
        if len(clean_line_ids) > 0:
            dataset = dataset.select(clean_line_ids)
        else:
            raise ValueError("This user does not exist or has no data.")
    return dataset


def load_stackoverflow(cache_dir="~/data", user_idx=0, return_full_dataset=False, split="train"):
    """Return the first 250 users if "return_full_dataset=True" ..."""
    os.makedirs(os.path.join(cache_dir, "cache"), exist_ok=True)
    if not return_full_dataset:
        path = os.path.join(cache_dir, "cache", f"stackoverflow_cache_{user_idx}.txt")
        try:
            with open(path, "r") as file:
                raw_texts = list(file)
        except FileNotFoundError:
            raw_texts = load_stackoverflow_tff(cache_dir=cache_dir, user_idx=user_idx, split=split)
            with open(path, "w") as file:
                for text in raw_texts:
                    file.write(text)
        return raw_texts
    else:
        text_collection = []
        for user_idx in range(250):
            raw_texts = load_stackoverflow_tff(cache_dir=cache_dir, user_idx=user_idx, split=split)
            text_collection += raw_texts
        return text_collection


def load_shakespeare(cache_dir="~/data", user_idx=0, return_full_dataset=False, split="train"):
    """Return the first 250 users if "return_full_dataset=True" ..."""
    os.makedirs(os.path.join(cache_dir, "cache"), exist_ok=True)
    if not return_full_dataset:
        path = os.path.join(cache_dir, "cache", f"shakespeare_cache_{user_idx}.txt")
        try:
            with open(path, "r") as file:
                raw_texts = list(file)
        except FileNotFoundError:
            raw_texts = load_shakespeare_tff(cache_dir=cache_dir, user_idx=user_idx, split=split)
            with open(path, "w") as file:
                for text in raw_texts:
                    file.write(text)
        return raw_texts
    else:
        text_collection = []
        for user_idx in range(250):
            raw_texts = load_shakespeare_tff(cache_dir=cache_dir, user_idx=user_idx, split=split)
            text_collection += raw_texts
        return text_collection


"""The following functions are adapted from tff at
https://github.com/tensorflow/federated/blob/610843c724740e1b041837cc93501b609fb05d8f/
tensorflow_federated/python/simulation/datasets/download.py#L31
and
https://github.com/tensorflow/federated/blob/610843c724740e1b041837cc93501b609fb05d8f/
tensorflow_federated/python/simulation/datasets/sql_client_data.py#L65
"""
# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import urllib
import urllib.parse

import lzma
import tqdm
import sqlite3


def url_basename(origin: str) -> str:
    origin_path = urllib.parse.urlparse(origin).path
    return origin_path.rsplit("/", maxsplit=1)[-1]


def _fetch_lzma_file(origin: str, filename: str):
    """Fetches a LZMA compressed file and decompresses on the fly."""
    # Read and decompress in approximately megabyte chunks.
    chunk_size = 2**20
    decompressor = lzma.LZMADecompressor()
    with urllib.request.urlopen(origin) as in_stream, open(filename, "wb") as out_stream:
        length = in_stream.headers.get("content-length")
        if length is not None:
            total_size = int(length)
        else:
            total_size = None
        download_chunk = in_stream.read(chunk_size)
        with tqdm.tqdm(total=total_size, desc=f"Downloading {url_basename(origin)}") as progbar:
            while download_chunk:
                progbar.update(len(download_chunk))
                out_stream.write(decompressor.decompress(download_chunk))
                download_chunk = in_stream.read(chunk_size)


def _load_sql_database(origin_url, cache_dir="~/data"):
    filename = url_basename(origin_url)
    local_filename = os.path.join(cache_dir, filename)
    extracted_filename, ext = os.path.splitext(local_filename)
    if os.path.exists(extracted_filename):
        return extracted_filename
    else:
        _fetch_lzma_file(origin_url, extracted_filename)
        return extracted_filename


def _fetch_client_id(database_filepath, user_idx, split_name=None):
    """Fetches the list of client_ids.
    Args:
      database_filepath: A path to a SQL database.
      user_idx: A numerical index to this user
      split_name: An optional split name to filter on. If `None`, all client ids
        are returned.
    Returns:
      An iterator of string client ids.
    """
    connection = sqlite3.connect(database_filepath)
    query = "SELECT DISTINCT client_id FROM client_metadata"
    if split_name is not None:
        query += f" WHERE split_name = '{split_name}'"
    query += ";"
    result = connection.execute(query)
    for idx, client_id in enumerate(result):
        if idx == user_idx:
            return client_id[0]
    else:
        raise ValueError(f"Given user idx {user_idx} larger than number of clients in database.")


TFF_URLS = {
    "stackoverflow": "https://storage.googleapis.com/tff-datasets-public/stackoverflow.sqlite.lzma",
    "shakespeare": "https://storage.googleapis.com/tff-datasets-public/shakespeare.sqlite.lzma",
}


def load_stackoverflow_tff(cache_dir="~/data", user_idx=0, split="train"):
    """Load the tensorflow federated stackoverflow dataset into pytorch."""
    if split == "validation":
        split_name = "heldout"
    elif split in ["train", "test"]:
        split_name = split
    else:
        raise ValueError(f"Split name {split} does not correspond to entries in this dataset.")
    db_name = _load_sql_database(TFF_URLS["stackoverflow"], cache_dir=cache_dir)
    client_id = _fetch_client_id(db_name, user_idx, split_name=split_name)
    query = (
        f"SELECT serialized_example_proto FROM examples WHERE client_id='{client_id}' and split_name='{split_name}';"
    )
    cursor = sqlite3.connect(db_name)
    result = cursor.execute(query)
    data = list(result)
    log.info(f"Now processing user {client_id} from tff database.")

    def parse_proto(tensor_proto):
        import tensorflow as tf  # wanted to circumvent this, but parsing the serialized data cleanly was difficult

        parse_spec = collections.OrderedDict(
            creation_date=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
            score=tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
            tags=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
            title=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
            tokens=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
            type=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
        )
        parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
        return parsed_features["tokens"].numpy().decode("ascii")

    raw_texts = []
    for proto_entry in data:
        raw_texts.append(parse_proto(proto_entry[0]))
    return raw_texts


def load_shakespeare_tff(cache_dir="~/data", user_idx=0, split="train"):
    """Load the tensorflow federated shakespeare dataset into pytorch."""
    if split == "validation":
        split = "test"
    if split in ["train", "test"]:
        split_name = split
    else:
        raise ValueError(f"Split name {split} does not correspond to entries in this dataset.")
    db_name = _load_sql_database(TFF_URLS["shakespeare"], cache_dir=cache_dir)
    client_id = _fetch_client_id(db_name, user_idx, split_name=split_name)
    query = (
        f"SELECT serialized_example_proto FROM examples WHERE client_id='{client_id}' and split_name='{split_name}';"
    )
    cursor = sqlite3.connect(db_name)
    result = cursor.execute(query)
    data = list(result)
    log.info(f"Now processing user {client_id} from tff database.")

    def parse_proto(serialized_proto_tensor):
        import tensorflow as tf  # wanted to circumvent this, but parsing the serialized data cleanly was difficult

        field_dict = {"snippets": tf.io.FixedLenFeature(shape=(), dtype=tf.string)}
        parsed_fields = tf.io.parse_example(serialized_proto_tensor, field_dict)
        return parsed_fields["snippets"].numpy().decode("ascii")

    raw_texts = []
    for proto_entry in data:
        raw_texts.append(parse_proto(proto_entry[0]))
    return raw_texts
