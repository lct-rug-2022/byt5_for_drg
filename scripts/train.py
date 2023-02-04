import random
from pathlib import Path

import numpy as np
import torch
import typer
import evaluate
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, EarlyStoppingCallback
)

from dataset_reader import (
    create_dataset,
    preprocess_dataset,
    create_collator
)
from predict import (
    run_prediction,
    GENERATION_LEN,
    N_BEAMS
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

app = typer.Typer(add_completion=False)

ROOT_FOLDER = Path(__file__).parent.parent.parent

DEV_METRIC = evaluate.load("chrf")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_chrf(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = DEV_METRIC.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"chrf": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


@app.command()
def main(
        base_model: str = typer.Option('google/byt5-small', help='ModelHub pretrained model to finetune'),
        learning_rate: float = typer.Option(1e-4, help='Learning Rate'),
        max_epochs: int = typer.Option(40, help='Number of Epochs'),
        batch_size: int = typer.Option(8, help='Batch size'),
        include_prefix: bool = typer.Option(False, is_flag=True, help='Whether to put language-specific task prefix'),
):
    model_name = f'clean_{base_model.rsplit("/", 1)[-1]}_{learning_rate}lr_{max_epochs}epochs'
    if include_prefix:
        model_name += '_prefix'

    model_save_dir = ROOT_FOLDER / 'models' / model_name

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    ds = create_dataset()
    ds = preprocess_dataset(ds, tokenizer, include_prefix=include_prefix)
    data_collator = create_collator(model, tokenizer)

    def compute_dev_metrics(eval_preds):
        return compute_chrf(eval_preds, tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=ROOT_FOLDER / 'checkpoints' / model_name,
        report_to='none',
        evaluation_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=max_epochs,
        save_strategy='epoch',
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=GENERATION_LEN,
        generation_num_beams=N_BEAMS,
        metric_for_best_model='eval_chrf',
        greater_is_better=True,
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_dev_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
    )

    trainer.train()
    trainer.save_model(str(model_save_dir / 'model'))

    for part in ['dev', 'test']:
        run_prediction(
            ds[part],
            trainer.model,
            tokenizer,
            model_save_dir,
            batch_size=batch_size,
            data_part=part,
            compute_metrics=True
        )


if __name__ == '__main__':
    app()

