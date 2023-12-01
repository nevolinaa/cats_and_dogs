import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.conv0 = nn.Conv2d(3, 128, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0)
        
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
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


def fit_model(model, loss_fn, optimizer, epochs, train_data_loader):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma = 0.6
    )

    loss_epochs_list = []
    acc_epochs_list = []

    for epoch in range(epochs):
        loss_val = 0
        acc_val = 0
        for sample in (pbar := tqdm(train_data_loader)):
            img, label = sample['img'], sample['label']
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

        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
        scheduler.step()
        loss_epochs_list += [loss_val/len(train_data_loader)]
        acc_epochs_list += [acc_val/len(train_data_loader)]
        print(loss_epochs_list[-1])
        print(acc_epochs_list[-1])
        
    return loss_epochs_list, acc_epochs_list