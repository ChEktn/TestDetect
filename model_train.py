import torch
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import time
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch import nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from torchvision.ops.boxes import nms


def draw_img_with_box(df_index, scale_percent=20):
    path_to_img = os.path.join('t_image',  df_train.loc[df_index, 'filename'])
    img = cv2.imread(path_to_img)
    boxes = df_train.loc[df_index, 'bbox']
    for box in boxes:
        cv2.rectangle(img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 0, 0), 5)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim)
    cv2.imwrite( "image1.png", img)


class MyDataset(Dataset):

    def __init__(self, dataframe, img_path):
        self.df = dataframe
        self.img_path = img_path

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_name = self.df.loc[index, 'filename']
        boxes = torch.Tensor(self.df.loc[index, 'bbox']).to(torch.float16)
        labels = torch.Tensor(self.df.loc[index, 'class']).to(torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        img = cv2.imread(os.path.join(self.img_path, img_name))
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float)
        return img, target

def create_model(num_classes, pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train(train_dataloader):
    model.train()
    running_loss = 0
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        images, targets = data[0], data[1]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"\tИтерация #{i} loss: {loss}")
    train_loss = running_loss/len(train_dataloader.dataset)
    return train_loss

def val(val_dataloader):
    running_loss = 0
    for data in val_dataloader:
        optimizer.zero_grad()
        images, targets = data[0], data[1]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        running_loss += loss.item()
    val_loss = running_loss/len(val_dataloader.dataset)
    return val_loss


if __name__=='__main__':
    df_train = pd.read_csv('train_t.csv',
                               names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])
    df_val = pd.read_csv('test_t.csv',
                             names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])


    df_train['bbox'] = df_train[['xmin', 'ymin', 'xmax', 'ymax']].apply(list, axis=1)
    df_val['bbox'] = df_val[['xmin', 'ymin', 'xmax', 'ymax']].apply(list, axis=1)

    df_train['class'] = df_train['class'].map({'table': 1})
    df_val['class'] = df_val['class'].map({'table': 1})

    df_train = df_train.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).groupby('filename', as_index=False).agg(list)
    df_val = df_val.drop(columns=['xmin', 'ymin', 'xmax', 'ymax']).groupby('filename', as_index=False).agg(list)

    draw_img_with_box(100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    model = create_model(num_classes=2).to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    train_dataset = MyDataset(df_train, 't_image')
    val_dataset = MyDataset(df_val, 't_image')

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    try:
        for epoch in range(10):
            start = time.time()
            train_loss = train(train_data_loader)
            val_loss = val(val_data_loader)
            scheduler.step()
            print(f"Эпоха #{epoch} train_loss: {train_loss}, val_loss: {val_loss}")
            end = time.time()
            print(f"Потрачено {round((end - start) / 60, 1)} минут на {epoch} эпоху")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print('Прервано пользователем')

    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn4.pth')