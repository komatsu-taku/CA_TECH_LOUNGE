import argparse
import os
from typing import List, DefaultDict
import json
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np


def make_label(dir: str, kind: str, image_path: str, labels: List[str], image_size: int):
    """
    yolo用のlabel (txtファイル)を作成
    """
    # 画像データのコピー
    name = os.path.basename(image_path)
    save_dir = os.path.join(dir, kind)
    # 画像のresize
    raw_image = Image.open(image_path)
    img_resized = raw_image.resize((image_size, image_size))
    img_resized.save(os.path.join(save_dir, name))
    # shutil.copyfile(image_path, os.path.join(save_dir, name))

    save_name = Path(name).stem + ".txt"
    save_path = Path(save_dir, save_name)
    
    # save label in txt file
    with open(save_path, "a") as f:
        f.writelines(labels)


def customize_bbox(image_path: str, label2num: DefaultDict, image_size: int):
    """
    Class > Label > xxx.txt から必要な情報を取ってくる
    yolo用にbbox情報を変更する
    [a, b, c, d] -> [a, b, h, w]
    前: 矩形の左上座標(a,b) & 右下座標(c,d)
    後] 矩形の左上座標(a,b) & 矩形の高さ,幅(h,w)

    Returns:
        ["label x y h w", ... , "label x y h w"]
    """
    dir = Path(image_path).parent
    basename = Path(image_path).stem
    dir = Path(dir, "Label")
    label_file = Path(dir, basename+".txt")

    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        H, W = image.shape
    else:
        H, W, C = image.shape

    with open(label_file, "r") as f:
        lines = f.readlines()
    
    output = []
    for line in lines:
        label_yolo = ""

        line = line.split()
        class_ = " ".join(line[:-4])

        a, b, c, d = map(float, line[-4:])
        # 縮尺を変更
        a = (a / W)
        b = (b / H)
        c = (c / W)
        d = (d / H)

        label = label2num[class_]

        # 座標値をyoloに合わせる
        h = d - b # 高さ
        w = c - a # 幅
        a = (a + c) / 2 # 中心のx座標
        b = (b + d) / 2 # 中心のy座標
        bbox_for_yolo = [a, b, h, w]

        label_yolo += str(label)

        for cor in bbox_for_yolo:
            label_yolo += " "
            label_yolo += str(cor)

        output.append(label_yolo)
    assert len(output) == len(lines)

    return output


def make_data_yolo(
        data: str,
        data_yolo: str,
        label2num: DefaultDict,
        image_size: int,):
    """
    yolo用のdata_yoloフォルダを作成する
    """
    image_set = ["train", "valid", "test"]

    for kind in image_set:
        path = os.path.join(data, f"{kind}.jsonl")
        df_kind = pd.read_json(path, orient="records", lines=True)
        
        W, H = df_kind.shape
        for idx in tqdm(range(W), desc=f"{kind}..."):
            info = df_kind.iloc[idx, :]

            image_path = info["image_path"]
            
            labels = customize_bbox(image_path, label2num, image_size)

            # 各ファイルを保存
            make_label(data_yolo, kind, image_path, labels, image_size)


def make_data_yaml(root: str, yaml: str, dir: str, label2num: DefaultDict):
    """
    yolo用のyamlファイルを作成
    contents:
        train: path/to/train
        val: path/to/valid
        test: path/to/test

        nc: number of class

        names: ["class1", "class2", ...., "classN"]
    """
    # ファイルがある場合は削除
    if os.path.isfile(yaml):
        os.remove(yaml)

    image_set = ["train", "val", "test"]

    content = ""
    for kind in image_set:
        content += f"{kind}: "
        if kind == "val":
            kind = "valid"
        path = os.path.join(root, dir, kind)
        content += path

        # 改行
        content += "\n"
    
    # class
    class_num = len(label2num)
    content += "\n"
    content += "nc: "
    content += f"{class_num}\n"
    content += "\n"

    names = list(label2num.keys())
    content += "names: "
    content += str(names)

    with open(yaml, "a") as f:
        f.write(content)


def main(args: argparse.Namespace):
    # yolo用のdataフォルダを作成
    data = args.data
    data_yolo = args.ydata
    image_size = args.image_size

    # foloderがある場合削除
    if os.path.isdir(data_yolo):
        shutil.rmtree(data_yolo)
    
    # フォルダの作成
    os.mkdir(data_yolo)
    for file in ["train", "valid", "test"]:
        os.mkdir(os.path.join(data_yolo, file))

    # label to num
    with open(args.label2num, "r") as f:
        label2num = json.load(f)

    # labelファイルの作成
    make_data_yolo(data, data_yolo, label2num, image_size)

    # yamlファイルの作成
    make_data_yaml(args.root, args.yaml, data_yolo, label2num)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", type=str, default="data/")
    parser.add_argument("-yd", "--ydata", type=str, default="data_yolo/")
    parser.add_argument("--label2num", type=str, default="./data/label2num.json")
    parser.add_argument("--yaml", type=str, default="data.yaml")
    parser.add_argument("--root", type=str, default="../")

    parser.add_argument("--image_size", type=int, default=224)

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
