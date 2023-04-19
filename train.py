import argparse
import os
import json
import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.utils import accuracy, AverageMeter

from utils.utils import fix_seed
from dataset import create_data_loader
from dataset.AnimalClass import AnimalDataset
from models.resnet import resnet50
from models.cnn import ConvNeXt
from test import test


class EarlyStopping:
    """
    Atrributes:
        patience(int): 何回まで値が減少しなくても続けるか、デフォルト7
        delta(float) : 前回のlossに加えてどれだけ良くなったら改善したとみなすか、デフォルト0
        save_dir(str): チェックポイントを保存するディレクトリ、デフォルトは"." (実行しているディレクトリ)
    """

    def __init__(
        self, patience: int = 7, delta: float = 0, save_dir: str = "."
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.save_dir = save_dir

        self.counter: int = 0
        self.early_stop: bool = False
        self.best_val_loss: float = np.Inf
        self.is_improve_val_loss: bool = False
        self.metric_log: str = ""

    def __call__(self, val_loss: float, net: nn.Module) -> str:
        if val_loss + self.delta < self.best_val_loss:
            log = f"({self.best_val_loss:.5f} --> {val_loss:.5f})"
            self._save_checkpoint(net)
            self.best_val_loss = val_loss
            self.counter = 0
            self.is_improve_val_loss = True
            return log

        self.counter += 1
        log = f"(> {self.best_val_loss:.5f} {self.counter}/{self.patience})"
        if self.counter >= self.patience:
            self.early_stop = True
        self.is_improve_val_loss = False
        return log

    def _save_checkpoint(self, net: nn.Module) -> None:
        # フォルダの作成
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        save_path = os.path.join(self.save_dir, "checkpoint.pt")
        torch.save(net.state_dict(), save_path)


def train(
    dataloader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn,
):
    total_loss: float = 0.0
    metric = AverageMeter()

    model.train()
    for data in tqdm(dataloader, "Train..."):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        loss.backward()
        total_loss += loss.item()
        
        acc1, acc5 = accuracy(outputs, labels, topk=(1,5))
        optimizer.step()

        metric.update(acc1.item(), outputs.size(0))

    return total_loss, metric.avg


def main(args: argparse.Namespace):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H%M%S")

    # フォルダの作成
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    fix_seed(args.seed, args.no_deterministic)

    # load data
    dataloader_dict = create_data_loader(
        batch_size=args.batch_size,
        image_size=args.image_size,
        only_test=False,
        root=args.data,
        is_transform=True,
    )

    with open(args.label2num, "r") as f:
        label2num = json.load(f)

    if args.model == "resnet50":
        model = resnet50(num_classes=len(label2num))
    elif args.model == "convnext":
        model = ConvNeXt(num_classes=len(label2num))

    loss_fn = nn.CrossEntropyLoss()
    metric = AverageMeter()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # save_dir
    run_name = f"{args.model}_{now_str}"
    save_dir = os.path.join(args.save_dir, run_name)
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, save_dir=save_dir
    )

    model.to(device)

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}]")

        for phase, dataloader in dataloader_dict.items():
            if phase == "Train":
                # train
                loss, metric = train(
                    dataloader,
                    model,
                    optimizer,
                    loss_fn,
                    )
            else:
                loss, metric = test(
                    dataloader,
                    model,
                    loss_fn,
                    device,
                )
            log = f"{phase}\t| Accuracy: {metric} Loss: {loss:.5f} "
            
            if phase == "Valid":
                early_stopping_log = early_stopping(loss, model)
                log += early_stopping_log
            
            if phase == "Test":
                if early_stopping.is_improve_val_loss:
                    early_stopping.metric_log = metric
            
            print(log)
        
        if early_stopping.early_stop:
            print("Early Stopping")
            model.load_state_dict(torch.load(os.path.join(save_dir, f"checkpoint.pt")))
            break

    print("Finish Training")
    print(f"Test_acc ; {early_stopping.metric_log}")

    if not os.path.isdir(args.checkpoints):
        os.makedirs(args.checkpoints)
    
    if not os.path.isdir(os.path.join(save_dir, args.checkpoints)):
        os.makedirs(os.path.join(save_dir, args.checkpoints))
    save_file = os.path.join(save_dir, args.checkpoints, f"{args.model}_{args.epochs}.pth")
    torch.save(model.state_dict(), save_file)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_deterministic", action="store_false")

    parser.add_argument("--data", type=str, default="./data/")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--label2num", type=str, default="data/label2num.json")

    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--iter", type=int, default=100)

    parser.add_argument("--save_dir", type=str, default="results/")

    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "convnext"])
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--save_weight_path", type=str, default="weights")

    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(parse_args())