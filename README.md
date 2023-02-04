# ByT5 for DRT semantic parsing

Authors: Ekaterina Garanina, Daragh Meehan, Qiankun Zheng


## Setup

Python:
```
git submodule init
git submodule update

python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Data:
```
wget https://pmb.let.rug.nl/releases/exp_data_4.0.0.zip
unzip exp_data_4.0.0.zip
```


## Training

To reproduce our experiments (training and evaluating ByT5 and mT5 with and without language-specific prefixes), run:

```
source env/bin/activate

python scripts/train.py --base-model=google/byt5-small
python scripts/train.py --base-model=google/byt5-small --include-prefix

python scripts/train.py --base-model=google/mt5-small
python scripts/train.py --base-model=google/mt5-small --include-prefix
```

On a single Nvidia V100 GPU, training time is ~12 hours for `byt5-small` and ~9 hours for `mt5-small`.

It is possible to change number of epochs (`--max-epochs`), learning rate (`--learning-rate`), and batch size (`--batch-size`).


## Prediction

Example command to run prediction:

```
source env/bin/activate
python scripts/predict.py --data-part=dev --model-dir=/data/models/byt5-small_0.0001lr_40epochs_prefix
```
The script will save the following files into the model folder:
* `predictions_[data-part].txt` - raw predictions with separators;
* `metrics_[data-part].csv` - csv table with gold annotations, generations, and SMATCH scores;
* `scores_[data-part].txt` - aggregated scores and percentage of invalid generations per language.