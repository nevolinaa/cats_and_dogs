import hydra
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from make_dataloader import Dataset2Class


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0)

        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 20)
        self.linear2 = nn.Linear(20, 2)

    def forward(self, x):
        out = self.conv0(x)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.act(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.act(out)

        out = self.adaptivepool(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)

        return out


def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def fit_model(cfg: DictConfig):
    model = ConvNet()
    loss_fn = nn.CrossEntropyLoss()
    train_ds_catsdogs = Dataset2Class(
        cfg.dataset.train_dogs_path, cfg.dataset.train_cats_path
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_ds_catsdogs,
        shuffle=cfg.train.shuffle,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        drop_last=cfg.train.drop_last,
    )

    OptimizerClass = hydra.utils.get_class(cfg.optimizer.optimizer_type)
    optimizer = OptimizerClass(model.parameters(), **cfg.optimizer.params)

    loss_epochs_list = []
    acc_epochs_list = []

    for epoch in range(cfg.general.epochs):
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(train_data_loader)):
            img, label = sample["img"], sample["label"]
            label = F.one_hot(label, 2).float()
            optimizer.zero_grad()

            pred = model(img)
            loss = loss_fn(pred, label)

            loss.backward()
            loss_item = loss.item()
            loss_val += loss_item

            optimizer.step()

            acc_current = accuracy(pred, label)
            acc_val += acc_current

        pbar.set_description(f"loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}")
        loss_epochs_list += [loss_val / len(train_data_loader)]
        acc_epochs_list += [acc_val / len(train_data_loader)]
        print(loss_epochs_list[-1])
        print(acc_epochs_list[-1])

    return loss_epochs_list, acc_epochs_list

def save_model(path, model):
    with open(path, "wb") as file:
        joblib.dump(model, file, compress=3)

if __name__ == "__main__":
    fit_model()
    save_model('model.joblib', model)
