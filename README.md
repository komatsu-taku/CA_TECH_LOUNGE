# CA_TECH_LOUNGE
This repository is for CA_TECH_LOUNGE. I worked on assignment 3 (Computer Vision②)

# Task1
## Setup

[![Python](https://img.shields.io/badge/python-3.8.10-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-3810/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.8.1-red?logo=pytorch)](https://download.pytorch.org/whl/torch_stable.html)

### Environment
* GeForce RTX 3080

### Requirenment
* Python 3.8.10
* pytorch 1.8.1

### Install our code
```bash
git clone --recursive git@github.com:komatsu-taku/CA_TECH_LOUNGE.git
```

### make data dir
```bash
cd CA_TECH_LOUNGE
mkdir data
```

### install package
```bash
pip install -r requirement.txt
```
When using GeFOrce 3090, the torch version must be changed as follows. For other environments, install the appropriate version of torch accordingly.
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Prepare Dataset
execute following code to preprocess dataset.  
All data I used is obtained [here](https://drive.google.com/file/d/18q9U-KkfoQWYqsTv2MU3KT34whfcOUVP/view?usp=share_link) instead of doing following codes.

### Download Dataset
> download dataet [here (kaggle)](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset) and move data/  

### Preprocess dataset
```python
python preprocess/create_jsonl.py --create_all_jsonl
```
After doing this code, make sure that following files are created [`data/all_data.jsonl`, `data/train.jsonl`, `data/valid.jsonl`, `data/test.jsonl`, `data/label_dict.json`]

## Training
At each epoch, execute train, valid, and test.
After training, weights are saved to `results/`
```python
python train.py \  
    --epoch 30
    --model resnet50
    --early_stopping_patience 5
    --image_size 224
    --batch_size 32
    --learning_rate 0.01 
```

## Inference
```python
python test.py \
    --model resnet50
    --checkpoint path/to/weights
```


# Task2
For simplicity, we did not create a separate folder, but ran it under the same folder.

### Preprocess
```python
python preprocess/prepare_yolo.py \
    --vdata data_yolo/
    --label2num ./data/label2num.json
    --yaml data.yaml
    --image_size 224
```

## Train
```python
python yolov5/train.py \
    --img 224 
    --data data.yaml 
    --weights yolov5s.pt 
    --batch-size 16
```

## Inference
After doing `train.py`, weights are obtained `yolo5/runs/train/expN/weights/bes.pt`, where N is shown in the log after train execution.
```python
python yolov5/detect.py \
    --source data_yolo/test 
    --weights path/to/weights 
    --conf 0.30
```
To test with a single image, run the following command
```python
python  yolov5/detect.py \
    --source path/to/image
    --weights path/to/weights
    --conf 0.30
```