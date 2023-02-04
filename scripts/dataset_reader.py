import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq

ROOT_FOLDER = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_FOLDER))
sys.path.append(str(ROOT_FOLDER / 'ud_boxer_repo'))

from ud_boxer_repo.ud_boxer.sbn import SBNGraph


SEED = 42

DATA_DIR = ROOT_FOLDER / 'exp_data_4.0.0'
PROBLEMATIC_DATA_FILES = [
    ROOT_FOLDER / 'ud_boxer_repo' / 'data' / 'misc' / 'cyclic_sbn_graph_paths.txt',
    ROOT_FOLDER / 'ud_boxer_repo' / 'data' / 'misc' / 'empty_sbn_docs.txt',
    ROOT_FOLDER / 'ud_boxer_repo' / 'data' / 'misc' / 'whitespace_in_ids.txt'
]

DATA_PARTS = ['train', 'dev', 'test']  # eval for English is only for complementary testing
TRAIN_PART = 'train'

LANGS = {
    'en': 'English',
    'de': 'German',
    'it': 'Italian',
    'nl': 'Dutch'
}

SPACE_REGEX = re.compile(r'\s+')
FILEPATH_PARTS_REGEX = re.compile(r'(p\d+)/(d\d+)/(\w\w\.drs\.sbn)')

MAX_SEQ_LEN = 512
LABEL_PAD_TOKEN_ID = -100


def create_problematic_data_list():
    problematic_data = set()
    for filepath in PROBLEMATIC_DATA_FILES:
        with open(filepath) as f:
            paths = f.read()

        for path_parts in FILEPATH_PARTS_REGEX.findall(paths):
            part, num, filename = path_parts
            problematic_data.add((part, num, filename))

    return problematic_data


def drg_to_penman(drg_str):
    try:
        penman = SBNGraph().from_string(drg_str).to_penman_string()
        return penman, None
    except Exception as e:
        return None, e


def format_drg(raw_drg):
    """
    Remove abundant spacing and all comments
    (including linking to raw text for nodes)
    """
    drg = '\n'.join(
        SPACE_REGEX.sub(' ', line.split('%')[0]).strip()
        for line in raw_drg.split('\n')
        if not line.startswith('%%%')
    ).strip()
    return drg


def preprocess_dataset(ds, tokenizer, include_prefix=False):
    def process(examples):
        texts = [
            f"Convert {LANGS[lang]} to DRG: {text}" if include_prefix else text
            for lang, text in zip(examples['lang'], examples['text'])
        ]

        model_inputs = tokenizer(texts, max_length=MAX_SEQ_LEN, truncation=True)
        labels = tokenizer(examples['drg'], max_length=MAX_SEQ_LEN, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    ds = ds.map(process, batched=True)
    return ds


def collect_stats(dataset_dict):
    stats = defaultdict(dict)
    for part, dataset in dataset_dict.items():
        for lang in LANGS.keys():
            stats[part][lang] = len(dataset.filter(lambda example: example['lang'] == lang))
    return stats


def create_dataset(data_parts=None):
    if data_parts is None:
        data_parts = DATA_PARTS

    data = {p: [] for p in data_parts}
    problematic_data = create_problematic_data_list()
    eval_ids = set()

    for lang in os.listdir(DATA_DIR):
        gold_folder = os.path.join(DATA_DIR, lang, 'gold')
        for part in data_parts:
            part_data = []

            with open(os.path.join(gold_folder, f'{part}.txt.raw')) as f:
                raw_sents = f.read().strip().split('\n')

            with open(os.path.join(gold_folder, f'{part}.txt.sbn')) as f:
                all_drg = f.read().strip()

            all_drg = all_drg.split('\n\n')
            assert len(raw_sents) == len(all_drg)

            for text, raw_drg in zip(raw_sents, all_drg):
                pmb_part, pmb_num, filename = FILEPATH_PARTS_REGEX.findall(raw_drg)[0]  # should be only one
                if (pmb_part, pmb_num, filename) not in problematic_data:
                    drg = format_drg(raw_drg)

                    # not all cases are in ud-boxer's problematic data lists
                    penman, error = drg_to_penman(drg)
                    if error is not None:
                        print(error, raw_drg, '', sep='\n')
                        continue

                    pmb_id = f'{pmb_part}/{pmb_num}'
                    part_data.append({
                        'pmb_id': pmb_id,
                        'lang': lang,
                        'text': text,
                        'drg': drg,
                        'penman': penman
                    })

                    if part != TRAIN_PART:
                        eval_ids.add(pmb_id)

            data[part].extend(part_data)

    ds = {part: Dataset.from_list(part_data) for part, part_data in data.items()}
    if TRAIN_PART in ds:
        ds[TRAIN_PART] = ds[TRAIN_PART].filter(lambda example: example['pmb_id'] not in eval_ids)

    ds = DatasetDict(ds)
    ds = ds.shuffle(seed=SEED)

    stats = collect_stats(ds)
    print('DATASET STATS:', json.dumps(stats, indent=2), sep='\n')

    return ds


def create_collator(model, tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=LABEL_PAD_TOKEN_ID
    )
