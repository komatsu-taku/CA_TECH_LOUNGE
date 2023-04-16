# CA_TECH_LOUNGE

## Setup
### Requirenment
> Python xx

### Install our code
```
git clone git@github.com:komatsu-taku/CA_TECH_LOUNGE.git
mkdir data
```

### Download Dataset
> download here (kaggle) and move data/


### Preprocess dataset
```
python preprocess/divide_train_data.py --create_all_jsonl
```
After doing this code, make sure that following files are created
> `data/all_data.jsonl`
> `data/train.jsonl`
> `data/valid.jsonl`
> `data/test.jsonl`
> `data/label_dict.json`

