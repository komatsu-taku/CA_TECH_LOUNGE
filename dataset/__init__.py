from typing import Dict
import os

import torchvision.transforms as transforms

from dataset.AnimalClass import AnimalDataset
from torch.utils.data import Dataset, DataLoader


def create_data_loader(
        batch_size: int,
        image_size: int,
        only_test: bool=False,
        root: str="data/",
        is_transform: bool=None,

) -> Dict[str, DataLoader]:
    """
    Args:
        batch_size: バッチサイズ
        image_size: 画像サイズ
        only_test: テストデータのみ作成
        root: dataディレクトリ
        is_transform: transformするかどうか
    """

    if is_transform:
        transform = transforms.Compose(
                [transforms.ToTensor(), # change to torch.tensor
                transforms.Resize((image_size, image_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean, distribution
    
    # test data
    test_dataset = AnimalDataset(root, "test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())

    if only_test:
        return {"Test", test_loader}
    
    train_dataset = AnimalDataset(root, "train", transform=transform)
    valid_dataset = AnimalDataset(root, "valid", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=os.cpu_count())

    dataloader_dict = {
        "Train": train_loader,
        "Valid": valid_loader,
        "Test": test_loader
    }

    return dataloader_dict
