import re
import json
import zstandard as zstd
import logging
import urllib.request
from typing import Dict
from urlextract import URLExtract
from datasets import Dataset, concatenate_datasets, DatasetDict
from datasets.features import Sequence, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig, AutoModelForMaskedLM

def stream_reddit_zst(path, max_lines=None):
    """
    Stream a Zstandard-compressed NDJSON Reddit file line by line.
    Each line is a JSON object (submission or comment).
    """
    with open(path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_stream = stream_reader.read().decode('utf-8').splitlines()
        for i, line in enumerate(text_stream):
            if max_lines and i >= max_lines:
                break
            yield json.loads(line)

def clean_reddit_text(text: str) -> str:
    """
    Clean Reddit-specific artifacts from text.
    """
    if text in ['[deleted]', '[removed]']:
        return ''
    text = re.sub(r'http\S+', '', text)  # remove links
    text = re.sub(r'>.*\n', '', text)    # remove quoted replies
    text = re.sub(r'\*+', '', text)      # remove markdown bold/italics
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # markdown links
    return text.strip()

def build_dataset_from_reddit(path, label: str, max_lines=10000):
    """
    Build a HuggingFace Dataset from a Reddit .zst file and assign a label.
    """
    examples = []
    for item in stream_reddit_zst(path, max_lines=max_lines):
        body = item.get("body") or item.get("selftext")
        if not body:
            continue
        text = clean_reddit_text(body)
        if not text:
            continue
        examples.append({
            "text": text,
            "label": label
        })
    return Dataset.from_list(examples)

def get_preprocessor(processor_type: str = None):
    url_ex = URLExtract()

    if processor_type is None:
        def preprocess(text):
            text = re.sub(r"@[A-Z,0-9]+", "@user", text)
            urls = url_ex.find_urls(text)
            for _url in urls:
                try:
                    text = text.replace(_url, "http")
                except re.error:
                    logging.warning(f're.error:\t - {text}\n\t - {_url}')
            return text

    elif processor_type == 'tweet_topic':
        def preprocess(tweet):
            urls = url_ex.find_urls(tweet)
            for url in urls:
                tweet = tweet.replace(url, "{{URL}}")
            tweet = re.sub(r"\b(\s*)(@[\S]+)\b", r'\1{\2@}', tweet)
            return tweet

    elif processor_type == 'reddit_topic':
        def preprocess(text):
            return clean_reddit_text(text)

    else:
        raise ValueError(f"unknown type: {processor_type}")

    return preprocess

def get_label2id(dataset: DatasetDict, label_name: str = 'label'):
    label_info = dataset[list(dataset.keys())[0]].features[label_name]
    while True:
        if type(label_info) is Sequence:
            label_info = label_info.feature
        else:
            assert type(label_info) is ClassLabel, f"Error at retrieving label information {label_info}"
            break
    return {k: n for n, k in enumerate(label_info.names)}


def load_model(model: str,
               task: str = 'sequence_classification',
               use_auth_token: bool = False,
               return_dict: bool = False,
               config_argument: Dict = None,
               model_argument: Dict = None,
               tokenizer_argument: Dict = None,
               model_only: bool = False):
    try:
        urllib.request.urlopen('http://google.com')
        no_network = False
    except Exception:
        no_network = True

    model_argument = {} if model_argument is None else model_argument
    model_argument.update({"use_auth_token": use_auth_token, "local_files_only": no_network})

    if return_dict or model_only:
        if task == 'sequence_classification':
            model = AutoModelForSequenceClassification.from_pretrained(model, return_dict=return_dict, **model_argument)
        elif task == 'token_classification':
            model = AutoModelForTokenClassification.from_pretrained(model, return_dict=return_dict, **model_argument)
        elif task == 'masked_language_model':
            model = AutoModelForMaskedLM.from_pretrained(model, return_dict=return_dict, **model_argument)
        else:
            raise ValueError(f'unknown task: {task}')
        return model

    config_argument = {} if config_argument is None else config_argument
    config_argument.update({"use_auth_token": use_auth_token, "local_files_only": no_network})
    config = AutoConfig.from_pretrained(model, **config_argument)

    tokenizer_argument = {} if tokenizer_argument is None else tokenizer_argument
    tokenizer_argument.update({"use_auth_token": use_auth_token, "local_files_only": no_network})
    tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_argument)

    model_argument.update({"config": config})
    if task == 'sequence_classification':
        model = AutoModelForSequenceClassification.from_pretrained(model, **model_argument)
    elif task == 'token_classification':
        model = AutoModelForTokenClassification.from_pretrained(model, **model_argument)
    elif task == 'masked_language_model':
        model = AutoModelForMaskedLM.from_pretrained(model, **model_argument)
    else:
        raise ValueError(f'unknown task: {task}')
    return config, tokenizer, model