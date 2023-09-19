# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
import torchvision


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.data_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.data_dir, self.label_dir)
        self.img_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        return img, self.label_dir

    def __len__(self):
        return len(self.img_list)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    writer = SummaryWriter("logs")
    train_set = torchvision.datasets.CIFAR10("./cifar10", True, transform=transforms.ToTensor(), download=True)
    test_set = torchvision.datasets.CIFAR10("./cifar10", False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    test_size = len(test_set)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    total_pass = 0
    max_accuracy = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    m = MyModule()
    m = m.to(device)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.0001)
    print("torch.cuda.is_available():", torch.cuda.is_available())

    for loop in range(100):
        print("--------- 第{}轮开始 ---------".format(loop))

        m.train()
        for data in train_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = m(imgs)
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_pass += 1
            if total_pass % 100 == 0:
                print("训练次数: {}, loss: {}".format(total_pass, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_pass)

        m.eval()
        total_accuracy = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = m(imgs)
                total_accuracy += (output.argmax(1) == targets).sum()
            new_accuracy = total_accuracy / test_size
            print("=======> 第{}轮正确率: {}".format(loop, new_accuracy))
            writer.add_scalar("test_accuracy", new_accuracy, loop)
            if new_accuracy > max_accuracy:
                max_accuracy = new_accuracy
                torch.save(m, "cifar10_best.pth")
                print("model saved, latest accuracy: {}".format(max_accuracy))

    writer.close()

