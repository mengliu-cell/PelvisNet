import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import pandas as pd
from sklearn.metrics import (confusion_matrix, mean_squared_error,
                             mean_absolute_error, r2_score)

from model import MEDCLIP_TRANSFORM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AgeDataset(Dataset):
    def __init__(self, imgs_ls, labels_ls, gender_ls, transform=MEDCLIP_TRANSFORM):
        self.imgs_ls   = imgs_ls
        self.labels    = labels_ls
        self.genders   = gender_ls
        self.transform = transform

    def __getitem__(self, item):
        age    = float(self.labels[item])
        gender = int(self.genders[item])
        image  = Image.open(self.imgs_ls[item]).convert('RGB')
        image  = self.transform(image)
        return image, age, gender

    def __len__(self):
        return len(self.labels)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_labels():
    df = pd.read_csv("/home/ubuntu/桌面/TXE-DR-数据-20240718/age_gender_6712.csv", header=None)
    ages    = df.iloc[:, 0]
    genders = df.iloc[:, 1]
    return ages, genders


def get_images():
    path = "/home/ubuntu/桌面/TXE-DR-数据-20240718/pelvis"
    images_ls = []
    for a in range(1, 8):
        sub_images = []
        for i in [1, 2, 3, 4, 5, 6, 7, 7.1, 7.2, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
            subs_images = []
            sorted_num  = []
            for item in glob.glob(os.path.join(path, f"{a}-{i}-*")):
                sorted_num.append(item.split("-")[-1].split(".")[0])
                sorted_num = sorted(sorted_num, key=int)
            for item in sorted_num:
                subs_images.append(f"{path}/{a}-{i}-{item}.png")
            sub_images += subs_images
        images_ls += sub_images
    images_ls = images_ls[::2]
    return images_ls


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def fit_epoch(model, train_dl, val_dl, loss_age, loss_sex, optim, scheduler, device):
    model.train()
    tr_age_preds, tr_age_true = [], []
    tr_run_loss = 0.0

    for batch in train_dl:
        imgs, ages, sexes = batch
        imgs  = imgs.to(device)
        ages  = ages.to(device).float()
        sexes = sexes.to(device).long()

        age_pred, sex_pred = model(imgs)
        loss = loss_age(age_pred, ages) + loss_sex(sex_pred, sexes)
        optim.zero_grad()
        loss.backward()
        optim.step()

        tr_run_loss += loss.item()
        with torch.no_grad():
            tr_age_preds.extend(age_pred.cpu().tolist())
            tr_age_true.extend(ages.cpu().tolist())

    scheduler.step()
    tr_loss = tr_run_loss / len(train_dl.dataset)

    tr_age_preds = np.array(tr_age_preds)
    tr_age_true  = np.array(tr_age_true)
    tr_mae  = mean_absolute_error(tr_age_true, tr_age_preds)
    tr_mse  = mean_squared_error(tr_age_true, tr_age_preds)
    tr_rmse = np.sqrt(tr_mse)
    tr_r2   = r2_score(tr_age_true, tr_age_preds)

    # ---- validation ----
    model.eval()
    val_age_preds, val_age_true = [], []
    val_sex_preds, val_sex_true = [], []
    val_run_loss = 0.0

    with torch.no_grad():
        for batch in val_dl:
            imgs, ages, sexes = batch
            imgs  = imgs.to(device)
            ages  = ages.to(device).float()
            sexes = sexes.to(device).long()

            age_pred, sex_pred = model(imgs)
            loss = loss_age(age_pred, ages) + loss_sex(sex_pred, sexes)
            val_run_loss += loss.item()

            val_age_preds.extend(age_pred.cpu().tolist())
            val_age_true.extend(ages.cpu().tolist())
            val_sex_preds.extend(torch.argmax(sex_pred, dim=1).cpu().tolist())
            val_sex_true.extend(sexes.cpu().tolist())

    val_loss = val_run_loss / len(val_dl.dataset)

    val_age_preds = np.array(val_age_preds)
    val_age_true  = np.array(val_age_true)
    mae  = mean_absolute_error(val_age_true, val_age_preds)
    mse  = mean_squared_error(val_age_true, val_age_preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(val_age_true, val_age_preds)

    correct = sum(p == t for p, t in zip(val_sex_preds, val_sex_true))
    sex_acc = correct / len(val_sex_true)
    cm      = confusion_matrix(val_sex_true, val_sex_preds)

    return (
        tr_loss, val_loss,
        tr_mae, tr_mse, tr_rmse, tr_r2,
        mae, mse, rmse, r2,
        sex_acc, cm,
        val_age_preds, val_age_true,
        val_sex_preds, val_sex_true,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def dynamic_plot_loss(losses):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Loss')
    ax.set_xlim(0, len(losses))
    ax.set_ylim(0, max(losses) * 1.1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    for i, _ in enumerate(losses):
        line.set_xdata(np.arange(i + 1))
        line.set_ydata(losses[:i + 1])
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.ioff()
    plt.show()
