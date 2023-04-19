import argparse
import json

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter

from utils.utils import fix_seed
from dataset.AnimalClass import AnimalDataset
from models.resnet import resnet50
from models.cnn import ConvNeXt

@torch.no_grad()
def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    device: torch.device,
):
    total_loss: float = 0.0
    metric = AverageMeter()

    model.eval()
    for data in tqdm(dataloader, "Test..."):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)

        total_loss += loss_fn(outputs, labels).item()
        acc1, acc5 = accuracy(outputs, labels, topk=(1,5))

        metric.update(acc1.item(), outputs.size(0))

    return total_loss, metric.avg #, metrics


def main(args: argparse.Namespace):
    fix_seed(args.seed, args.no_deterministic)

    # load data
    transform = transforms.Compose(
            [transforms.ToTensor(), # change to torch.tensor
             transforms.Resize((args.image_size, args.image_size)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean, distribution
    
    test_dataset = AnimalDataset(args.data, "test", transform=transform)
    test_dataloader = DataLoader(test_dataset)

    # label
    with open(args.label2num, "r") as f:
        label2num = json.load(f)
    
    if args.model == "resnet50":
        model = resnet50(num_classes=len(label2num))
    elif args.model == "convnext":
        model = ConvNeXt(num_classes=len(label2num))

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    loss, metrics = test(test_dataloader, model, loss_fn, device)

    print(f"Test\t| Accuracy: {metrics} Loss: {loss:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_deterministic", action="store_false")

    parser.add_argument("--data", type=str, default="./data/")

    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--label2num", type=str, default="data/label2num.json")
    parser.add_argument("-m", "--model", type=str, choices=["resnet50", "convnext"], default="resnet50")
    parser.add_argument("-c", "--checkpoint", type=str)


    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())