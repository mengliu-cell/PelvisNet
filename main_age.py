import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, dataset
from sklearn.model_selection import KFold
from tqdm import tqdm

from model import AgeGenderModel
from train import AgeDataset, get_labels, get_images, fit_epoch, dynamic_plot_loss


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCH    = 60
    K        = 5
    best_mae = float('inf')
    best_acc = 0.0

    MEDCLIP_CKPT = None  # e.g. './pretrained/medclip-vit'

    ages, genders = get_labels()
    images_ls     = get_images()

    bins = [
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
        (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 35),
    ]
    train_id, test_id = [], []
    for lo, hi in bins:
        idx = ages[(ages >= lo) & (ages < hi)].index.tolist()
        cut = int(len(idx) * 0.8)
        train_id.extend(idx[:cut])
        test_id.extend(idx[cut:])

    train_images = [images_ls[i] for i in train_id]
    train_age    = [ages[i]      for i in train_id]
    train_gender = [genders[i]   for i in train_id]

    train_val_data = AgeDataset(train_images, train_age, train_gender)
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    fold = 1
    df   = pd.DataFrame(columns=[
        "fold", "epoch",
        "train_mae", "val_mae", "train_rmse", "val_rmse",
        "train_mse", "val_mse", "train_r2", "val_r2",
        "train_loss", "val_loss", "val_sex_acc",
    ])

    for train_index, val_index in kf.split(train_val_data):
        model = AgeGenderModel(medclip_checkpoint=MEDCLIP_CKPT).to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        loss_age  = nn.MSELoss()
        loss_sex  = nn.CrossEntropyLoss()

        train_ds = dataset.Subset(train_val_data, train_index)
        val_ds   = dataset.Subset(train_val_data, val_index)
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        row_lists = {k: [] for k in df.columns}
        losses = []

        for epoch in tqdm(range(EPOCH), desc=f"Fold {fold}"):
            (tr_loss, val_loss,
             tr_mae, tr_mse, tr_rmse, tr_r2,
             mae, mse, rmse, r2,
             sex_acc, cm,
             val_age_preds, val_age_true,
             val_sex_preds, val_sex_true) = fit_epoch(
                model, train_dl, val_dl, loss_age, loss_sex, optimizer, scheduler, device
            )

            save_dir = "/home/ubuntu/桌面/SAM2PATH/pelvis/age_gender_results_medclip"
            os.makedirs(save_dir, exist_ok=True)
            pd.DataFrame({
                "id_index":        list(val_index),
                "val_pred_age":    val_age_preds.tolist(),
                "val_true_age":    val_age_true.tolist(),
                "val_pred_gender": val_sex_preds,
                "val_true_gender": val_sex_true,
            }).to_csv(os.path.join(save_dir, f"fold_{fold}_epoch_{epoch+1}.csv"), index=False)

            print(f"fold:{fold} epoch:{epoch+1}/{EPOCH}  "
                  f"tr_loss:{tr_loss:.4f}  val_loss:{val_loss:.4f}  "
                  f"val_MAE:{mae:.3f}  val_sex_acc:{sex_acc:.3f}")

            losses.append(tr_loss)
            dynamic_plot_loss(losses)

            for k, v in zip(df.columns, [
                fold, epoch+1,
                tr_mae, mae, tr_rmse, rmse,
                tr_mse, mse, tr_r2, r2,
                tr_loss, val_loss, sex_acc,
            ]):
                row_lists[k].append(v)

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(),
                           "/home/ubuntu/桌面/SAM2PATH/pelvis/best_age_model_medclip.pth")
            if sex_acc > best_acc:
                best_acc = sex_acc
                torch.save(model.state_dict(),
                           "/home/ubuntu/桌面/SAM2PATH/pelvis/best_sex_model_medclip.pth")

        df = pd.concat([df, pd.DataFrame(row_lists)], ignore_index=True)
        fold += 1

    df.to_csv("/home/ubuntu/桌面/SAM2PATH/pelvis/results_medclip.csv", index=False)


if __name__ == '__main__':
    main()
