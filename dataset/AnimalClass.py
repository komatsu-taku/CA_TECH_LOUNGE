import os
from typing import Optional, Callable

from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AnimalDataset(Dataset):
    """
    root(str) : データセットのディレクトリ root/data/ が存在することを仮定
    image_set(str): データセットの種類 train / valid / test
    transform(Collable) : 画像の変換
    """
    def __init__(
            self, root: str, image_set: str="train", transform: Optional[Callable]=None
        ):
        super().__init__()
        
        self.root = root
        self.image_set = image_set
        self.transform = transform

        if self.image_set == "test":
            self.data_dir = os.path.join(self.root, self.image_set)
            self.jsonl_file = os.path.join(self.root, self.image_set+".jsonl")
        else:
            self.data_dir = os.path.join(self.root, "train")
            self.jsonl_file = os.path.join(self.root, self.image_set+".jsonl")

        df = pd.read_json(self.jsonl_file, orient="record", lines=True)
        self.images = df["image_path"].to_list()
        self.labels = df["label"].to_list()

        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label)
    
    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    transform = transforms.Compose(
            [transforms.ToTensor(), # change to torch.tensor
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean, distribution
    
    train_dataset = AnimalDataset("./data/", image_set="train", transform=transform)
    valid_dataset = AnimalDataset("./data/", image_set="valid", transform=transform)
    test_dataset = AnimalDataset("./data/", image_set="test", transform=transform)

    # test
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    test_image, test_label = train_dataset[100]
    test_image = test_image / 2 + 0.5 # demnormalize
    plt.imshow(np.transpose(test_image, (1,2,0)))

    with open("data/num2label.json", "r") as f:
        num2label = json.load(f)
    
    title = str(test_label.item()) + ": " + num2label[str(test_label.item())]
    plt.title(title)
    plt.savefig("fig/test_image.png")

    from torch.utils.data import DataLoader
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())
    