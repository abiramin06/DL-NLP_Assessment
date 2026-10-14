import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of samples:", len(y))

X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

class CsvDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(
            features, dtype=torch.float32
        )
        self.y = torch.tensor(
            labels, dtype=torch.long
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = CsvDataset(X_train, y_train)
val_dataset = CsvDataset(X_val, y_val)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Number of training batches:", len(train_loader))
print("Number of validation batches:", len(val_loader))

features, labels = next(iter(train_loader))

print("Batch feature shape:", features.shape)
print("Batch label shape:", labels.shape)


class MLP(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = MLP()

total_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nModel Architecture:",model)
print("\nTrainable Parameters:", total_params)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 40
patience = 5

best_val_accuracy = 0.0
best_epoch = 0
epochs_without_improvement = 0

train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), "task2/best_model.pth")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered!")
        break

print("\nBest validation accuracy:", round(best_val_accuracy, 4))
print("Best epoch:", best_epoch)

model.load_state_dict(
    torch.load(
        "task2/best_model.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()
reloaded_model = MLP()
reloaded_model.load_state_dict(
    torch.load(
        "task2/best_model.pth",
        map_location="cpu",
        weights_only=True
    )
)
reloaded_model.eval()

plt.plot(range(1, len(train_losses) + 1), train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("PyTorch MLP Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("task2/training_loss.png")
plt.show()

reloaded_model = MLP()
reloaded_model.load_state_dict(
    torch.load("task2/best_model.pth", map_location="cpu",
               weights_only=True)
)
reloaded_model.eval()
model.eval()

def get_predictions(current_model, loader):
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            outputs = current_model(features)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_predictions), np.array(all_labels)


original_preds, true_labels = get_predictions(model, val_loader)
reloaded_preds, _ = get_predictions(reloaded_model, val_loader)

original_accuracy = accuracy_score(true_labels, original_preds)
reloaded_accuracy = accuracy_score(true_labels, reloaded_preds)

print("Original model accuracy:", round(original_accuracy, 4))
print("Reloaded model accuracy:", round(reloaded_accuracy, 4))
print("Predictions identical:", np.array_equal(original_preds, reloaded_preds))

logistic_model = LogisticRegression(max_iter=2000)
logistic_model.fit(X_train, y_train)

logistic_preds = logistic_model.predict(X_val)
logistic_accuracy = accuracy_score(y_val, logistic_preds)

print("\nLogistic Regression accuracy:", round(logistic_accuracy, 4))

print("\nLogistic Regression Report:",classification_report(y_val, logistic_preds))