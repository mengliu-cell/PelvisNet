
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MedCLIP-main', 'MedCLIP-main'))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score,
    accuracy_score, classification_report,
)
from tqdm import tqdm

# reuse model + data helpers from train.py
from model import AgeGenderModel, MEDCLIP_TRANSFORM
from train import AgeDataset, get_labels, get_images


# ---------------------------------------------------------------------------
# Config — edit these paths before running
# ---------------------------------------------------------------------------
CHECKPOINT  = "/home/ubuntu/SAM2PATH/pelvis/best_age_model_medclip.pth"
MEDCLIP_CKPT = None          # same value as in train.py
SAVE_DIR     = "/home/ubuntu/SAM2PATH/pelvis/test_results_medclip"
BATCH_SIZE   = 32
NUM_WORKERS  = 4
# ---------------------------------------------------------------------------


def build_test_loader():
    ages, genders = get_labels()
    images_ls     = get_images()

    bins = [
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
        (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
        (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24),
        (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 35),
    ]
    test_id = []
    for lo, hi in bins:
        idx = ages[(ages >= lo) & (ages < hi)].index.tolist()
        cut = int(len(idx) * 0.8)
        test_id.extend(idx[cut:])

    test_images  = [images_ls[i] for i in test_id]
    test_ages    = [ages[i]      for i in test_id]
    test_genders = [genders[i]   for i in test_id]

    ds = AgeDataset(test_images, test_ages, test_genders, transform=MEDCLIP_TRANSFORM)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    return dl, test_id


@torch.no_grad()
def evaluate(model, test_dl, device):
    model.eval()
    age_preds, age_true = [], []
    sex_preds, sex_true = [], []

    for imgs, ages, sexes in tqdm(test_dl, desc="Testing"):
        imgs  = imgs.to(device)
        ages  = ages.to(device).float()
        sexes = sexes.to(device).long()

        age_pred, sex_pred = model(imgs)

        age_preds.extend(age_pred.cpu().tolist())
        age_true.extend(ages.cpu().tolist())
        sex_preds.extend(torch.argmax(sex_pred, dim=1).cpu().tolist())
        sex_true.extend(sexes.cpu().tolist())

    return (np.array(age_preds), np.array(age_true),
            np.array(sex_preds), np.array(sex_true))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dl, test_id = build_test_loader()
    print(f"Test samples: {len(test_dl.dataset)}")

    model = AgeGenderModel(medclip_checkpoint=MEDCLIP_CKPT).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {CHECKPOINT}")

    age_preds, age_true, sex_preds, sex_true = evaluate(model, test_dl, device)

    # --- age regression metrics ---
    mae  = mean_absolute_error(age_true, age_preds)
    mse  = mean_squared_error(age_true, age_preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(age_true, age_preds)

    # --- sex classification metrics ---
    acc = accuracy_score(sex_true, sex_preds)
    cm  = confusion_matrix(sex_true, sex_preds)

    print("\n===== Test Results =====")
    print(f"Age  — MAE: {mae:.4f}  RMSE: {rmse:.4f}  MSE: {mse:.4f}  R²: {r2:.4f}")
    print(f"Sex  — Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(sex_true, sex_preds, target_names=["Female", "Male"]))

    # --- save raw predictions ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.DataFrame({
        "id_index":        test_id,
        "pred_age":        age_preds.tolist(),
        "true_age":        age_true.tolist(),
        "pred_gender":     sex_preds.tolist(),
        "true_gender":     sex_true.tolist(),
        "age_error":       (age_preds - age_true).tolist(),
    })
    out_csv = os.path.join(SAVE_DIR, "test_predictions.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nPredictions saved to: {out_csv}")

    # --- save summary metrics ---
    summary = pd.DataFrame([{
        "checkpoint": CHECKPOINT,
        "n_test":     len(test_id),
        "age_mae":    mae,
        "age_rmse":   rmse,
        "age_mse":    mse,
        "age_r2":     r2,
        "sex_acc":    acc,
    }])
    summary_csv = os.path.join(SAVE_DIR, "test_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Summary saved to: {summary_csv}")


if __name__ == '__main__':
    main()
