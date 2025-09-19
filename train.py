import torch 
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_create_tokenizer(tokenizer_path, ds, lang):
    tokenizer_path = Path(tokenizer_path)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            ,min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang),
            trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset()

    ## Build tokenizer