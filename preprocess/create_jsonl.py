import os
import json
import argparse
from glob import glob
from typing import List


from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


def get_data_path(args: argparse.Namespace):
    """
    学習データおよびテストデータのディレクトリパス および
    各クラスのディレクトリのリストを返す
    """
    train_dir = os.path.join(args.data, args.train_dpath)
    test_dir = os.path.join(args.data, args.test_dpath)
    
    all_train_subdir = glob(train_dir+"*")
    all_test_subdir = glob(test_dir+"*")
    assert len(all_train_subdir) == 80

    return train_dir, test_dir, all_train_subdir, all_test_subdir


def check_number_classes(train_class_dir: List, test_class_dir: List):
    """
    学習データおよびテストデータのクラス数を確認し、出力する関数
    """
    train_classes = [os.path.basename(pp) for pp in train_class_dir]
    test_classes = [os.path.basename(pp) for pp in test_class_dir]

    print("number of classes of train data: ", len(train_classes))
    print("number of classes of test  data: ", len(test_classes))


def plot_number_each_classes(train_class_dir, test_class_dir):
    """
    train / test データに含まれる
    各クラスの画像枚数を出力する
    """
    train_image_counts={os.path.basename(pp):[len(glob(os.path.join(pp, "*.jpg")))] for pp in train_class_dir}
    test_image_counts={os.path.basename(pp):[len(glob(os.path.join(pp, "*.jpg")))] for pp in test_class_dir}

    train_data_df = pd.DataFrame(train_image_counts, index=["train"]).transpose()
    test_data_df = pd.DataFrame(test_image_counts, index=["test"]).transpose()
    all_data_df=train_data_df.copy()
    all_data_df["test"]=test_data_df
    print(all_data_df.head())

    plt.figure(figsize=(20,16))
    all_data_df.plot.bar()
    plt.savefig("   fig/number_each_classes.png")


def create_data_jsonl(args, train_dir, test_dir):
    """
    各データの情報をjson形式で保存したファイルを作成
    "image_path": image_path
    "class_" : class(str)
    "label" : 各クラスの該当数字(0~20)
    "bbox" : [float, float, float, float]
    "kind" : "train"or"test"
    """
    # remove file if exist
    if os.path.isfile(args.json_path):
        os.remove(args.json_path)

    # label_dict
    with open(args.label_dict, "r") as f:
        label2num = json.load(f)
    
    # train
    for dir in tqdm(train_dir, desc="create train..."):
        image_files = glob(os.path.join(dir, "*.jpg"))
        for image_file in image_files:
            mp = {}
            # print(image_file) : data/train/Goose/34a459289ab236d4.jpg
            im_name = os.path.basename(image_file).split(".")[0]
            label_path = os.path.join(dir, "Label", im_name+".txt")
            with open(label_path, "r") as f:
                info = f.readline()
                info = info.split()
            label = os.path.basename(os.path.dirname(image_file))
            bbox = list(map(float, info[-4:]))

            # make json (each image)
            mp["image_path"] = image_file
            mp["class_"] = label
            mp["label"] = label2num[label]
            mp["bbox"] = bbox
            mp["kind"] = "train"

            with open(args.json_path, "a") as f:
                enc = json.dumps(mp)
                f.write(enc)
                f.write("\n")

    # test
    for dir in tqdm(test_dir, desc="create test..."):
        image_files = glob(os.path.join(dir, "*.jpg"))
        for image_file in image_files:
            mp = {}
            im_name = os.path.basename(image_file).split(".")[0]
            label_path = os.path.join(dir, "Label", im_name+".txt")
            with open(label_path, "r") as f:
                info = f.readline()
                info = info.split()
            label = os.path.basename(os.path.dirname(image_file))
            bbox = list(map(float, info[-4:]))

            # make json (each image)
            mp["image_path"] = image_file
            mp["class_"] = label
            mp["label"] = label2num[label]
            mp["bbox"] = bbox
            mp["kind"] = "test"

            with open(args.json_path, "a") as f:
                enc = json.dumps(mp)
                f.write(enc)
                f.write("\n")


def devide_data(args):
    """
    train / valid / test ごとのjsonlファイルを作成
    train : val = 8 : 2 となるようにtrainデータを分割
    """
    for file in ["data/train.jsonl", "data/valid.jsonl", "data/test.jsonl"]:
        if os.path.isfile(file):
            os.remove(file)

    all_data = pd.read_json(args.json_path, orient="record", lines=True)

    test_df = all_data[all_data["kind"] == "test"]
    train_df = all_data[all_data["kind"] == "train"]

    # shuffle
    train_df = train_df.sample(frac=1, random_state=42)
    
    # divide
    W, H = train_df.shape
    index = int(W * args.ratio)

    div_train_df = train_df.iloc[:index, :]
    div_valid_df = train_df.iloc[index:, :]
    assert all_data.shape[0] == test_df.shape[0]+div_train_df.shape[0]+div_valid_df.shape[0]

    test_df.to_json("data/test.jsonl", orient="records", force_ascii=False, lines=True)
    div_train_df.to_json("data/train.jsonl", orient="records", force_ascii=False, lines=True)
    div_valid_df.to_json("data/valid.jsonl", orient="records", force_ascii=False, lines=True)


def create_label_dict(args, train_class_dir: str):
    """
    各クラスを0~79に割り当てた辞書を作成
    """
    if os.path.isfile(args.label_dict):
        os.remove(args.label_dict)
    
    label2num = {}
    for idx, dir in enumerate(train_class_dir):
        name = os.path.basename(dir)
        label2num[name] = idx
    
    with open(args.label_dict, "a") as f:
        json.dump(label2num, f, indent=2, ensure_ascii=False)

def main(args: argparse.Namespace):
    # get all data path
    train_dir, test_dir, all_train_class_dir, all_test_class_dir = get_data_path(args)

    # confirm number of classes of train-data and test-data
    if args.check:
        check_number_classes(all_train_class_dir, all_test_class_dir)
        plot_number_each_classes(all_train_class_dir, all_test_class_dir)
    
    # class to label
    create_label_dict(args, all_train_class_dir)

    # load all data path
    if args.create_all_jsonl:
        create_data_jsonl(args, all_train_class_dir, all_test_class_dir)
    
    # divide data train -> train / valid test -> test
    devide_data(args)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", type=str, default="data/")
    parser.add_argument("-dtr", "--train_dpath", type=str, default="train/")
    parser.add_argument("-dte", "--test_dpath", type=str, default="test/")
    parser.add_argument("-ch", "--check", action="store_true")
    parser.add_argument("-j", "--json_path", type=str, default="data/all_data.jsonl")
    parser.add_argument("--label_dict", type=str, default="data/label_dict.json")
    parser.add_argument("--create_all_jsonl", action="store_true")
    parser.add_argument("-r", "--ratio", type=float, default=0.8)

    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())