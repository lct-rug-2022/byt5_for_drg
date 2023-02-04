import sys
import json
import random
from pathlib import Path

import typer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer
)

ROOT_FOLDER = Path(__file__).parent.parent
sys.path.append(str(ROOT_FOLDER))
sys.path.append(str(ROOT_FOLDER / 'ud_boxer_repo'))

from ud_boxer_repo.ud_boxer.helpers import (
    smatch_score,
    _RELEVANT_ITEMS,
    _KEY_MAPPING
)
from dataset_reader import (
    drg_to_penman,
    create_dataset,
    preprocess_dataset,
    create_collator,
    MAX_SEQ_LEN
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

prediction_app = typer.Typer(add_completion=False)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GENERATION_LEN = MAX_SEQ_LEN
N_BEAMS = 3

SMATCH_KEYS = [_KEY_MAPPING[item] for item in _RELEVANT_ITEMS]
PREDICTION_SEP = '\n' + '='*30 + '\n'


def compute_smatch(y_true, y_pred, model_dir):
    tempgold = model_dir / 'tempgold'
    temppred = model_dir / 'temppred'

    with open(tempgold, "w") as gold_f:
        gold_f.write(y_true)

    with open(temppred, "w") as pred_f:
        pred_f.write(y_pred)

    scores = smatch_score(tempgold, temppred)
    return scores


def collect_test_metrics(test_dataset, preds, model_dir):
    lines = []

    for y_true_example, y_pred_drg in zip(test_dataset, preds):
        line = {
            'pmb_id': f"{y_true_example['lang']}/{y_true_example['pmb_id']}",
            'lang': y_true_example['lang'],
            'raw': y_true_example['text'],
            'gold': y_true_example['drg'],
            'pred': y_pred_drg
        }

        y_pred_penman, error = drg_to_penman(y_pred_drg)
        if error is not None:  # generated invalid / cyclic DRG
            line['error'] = error
            lines.append(line)
            continue

        smatch = compute_smatch(y_true_example['penman'], y_pred_penman, model_dir)
        for k, v in smatch.items():
            line[k] = v

        lines.append(line)

    df = pd.DataFrame(lines)
    return df


def compute_and_write_test_metrics(test_dataset, preds, model_dir, data_part):
    smatch_scores = {}
    invalid_proportions = {}

    metrics_df = collect_test_metrics(test_dataset, preds, model_dir)
    metrics_df.to_csv(model_dir / f'metrics_{data_part}.csv', index=False)

    for lang in metrics_df.lang.unique():
        lang_df = metrics_df.loc[metrics_df.lang == lang]

        invalid_proportion = sum(pd.notna(lang_df.error)) / lang_df.shape[0]
        invalid_proportions[lang] = round(invalid_proportion * 100, 1)

        lang_df_zeroes = lang_df.fillna(0)
        smatch_scores[lang] = {
            k: round(lang_df_zeroes[k].mean() * 100, 1)
            for k in SMATCH_KEYS
        }

    with open(model_dir / f'scores_{data_part}.txt', 'w') as f:
        f.write(f'SMATCH:\n{json.dumps(smatch_scores, indent=2)}\n\n')
        f.write(f'Invalid:\n{json.dumps(invalid_proportions, indent=2)}')


def run_prediction(dataset, model, tokenizer, model_dir, batch_size, data_part, compute_metrics=True):
    model.eval()

    collator = create_collator(model, tokenizer)
    tokens = dataset.remove_columns(['pmb_id', 'lang', 'text', 'drg', 'penman', 'labels'])  # a bit of hardcoding
    test_dataloader = DataLoader(tokens, batch_size=batch_size, collate_fn=collator)

    all_preds = []
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds = model.generate(**batch, num_beams=N_BEAMS, max_length=GENERATION_LEN)
        decoded_preds = tokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        all_preds.extend(decoded_preds)

    with open(model_dir / f'predictions_{data_part}.txt', 'w') as f:
        f.write(PREDICTION_SEP.join(all_preds))

    if compute_metrics:
        compute_and_write_test_metrics(
            test_dataset=dataset,
            preds=all_preds,
            model_dir=model_dir,
            data_part=data_part
        )


@prediction_app.command()
def run_test_prediction(
        data_part: str = typer.Option('test', help='Data part to predict on'),
        model_dir: str = typer.Option(..., help='Path to model for inference'),
        batch_size: int = typer.Option(8, help='Batch size'),
        include_prefix: bool = typer.Option(False, is_flag=True, help='Whether to put language-specific task prefix'),
):
    if '_prefix' in model_dir:  # model 100% trained with prefixes
        include_prefix = True

    model_dir = Path(model_dir)
    path_to_model = str(model_dir / 'model')
    model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    ds = create_dataset(data_parts=[data_part])[data_part]
    ds = preprocess_dataset(ds, tokenizer, include_prefix=include_prefix)

    run_prediction(
        ds,
        model,
        tokenizer,
        model_dir,
        batch_size=batch_size,
        data_part=data_part,
        compute_metrics=True
    )


if __name__ == '__main__':
    prediction_app()
