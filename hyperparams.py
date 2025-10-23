import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time
import pandas as pd
import importlib

# ======================================================
# CONFIGURATION
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/path/to/dataset"  # change to your dataset path
ModelModule = importlib.import_module("model")  # imports model.py dynamically
ModelClass = getattr(ModelModule, "DenseNetMSSA")  # adjust if different class name

# Hyperparameter search space (aligned with your Overleaf table)
configs = [
    {"name": "Baseline", "lr": 1e-3, "batch_size": 64, "dropout": 0.0},
    {"name": "Tuned (Ours)", "lr": 1e-4, "batch_size": 32, "dropout": 0.3},
    {"name": "Aggressive Reg.", "lr": 1e-4, "batch_size": 32, "dropout": 0.5},
    {"name": "Large Batch", "lr": 1e-3, "batch_size": 128, "dropout": 0.1},
    {"name": "Small Batch", "lr": 1e-4, "batch_size": 16, "dropout": 0.3},
]

num_classes = 10  # adjust per dataset (e.g., # of wildlife species)
num_epochs = 10  # or 15 if you want stable convergence

# Energy model assumptions (from Overleaf section)
GPU_POWER_KW = 0.4  # average A100 draw
PUE = 1.2
CARBON_INTENSITY = 0.37  # kg CO2e/kWh (EU grid mix)


# ======================================================
# DATASET PIPELINE
# ======================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)


# ======================================================
# TRAINING FUNCTION
# ======================================================

def train_model(cfg):
    print(f"\n=== Training Config: {cfg['name']} ===")
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = ModelClass(dropout=cfg["dropout"], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    scaler = torch.cuda.amp.GradScaler()  # AMP for energy efficiency

    best_acc, total_time = 0, 0
    start_time = time()

    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)

    total_time = (time() - start_time) / 3600  # hours
    energy_kwh = GPU_POWER_KW * total_time * PUE
    co2_kg = energy_kwh * CARBON_INTENSITY

    print(f"Accuracy: {best_acc:.2f}% | Time: {total_time:.2f}h | Energy: {energy_kwh:.2f} kWh | CO₂: {co2_kg:.2f} kg")

    return {
        "Config": cfg["name"],
        "LR": cfg["lr"],
        "Batch": cfg["batch_size"],
        "Dropout": cfg["dropout"],
        "Accuracy (%)": round(best_acc, 2),
        "GPU Hours": round(total_time, 2),
        "Energy (kWh)": round(energy_kwh, 2),
        "CO₂e (kg)": round(co2_kg, 2),
    }


# ======================================================
# RUN EXPERIMENTS
# ======================================================

results = []
for cfg in configs:
    results.append(train_model(cfg))

# ======================================================
# SAVE RESULTS
# ======================================================

df = pd.DataFrame(results)
df.to_csv("hyperparam_tuning_results.csv", index=False)
print("\n=== Final Hyperparameter Comparison ===")
print(df)
