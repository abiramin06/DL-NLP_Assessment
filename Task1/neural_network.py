import numpy as np
import torch
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

print("ReLU:", relu(np.array([-2, 0, 3])))
print("Sigmoid:", sigmoid(np.array([-2, 0, 3])))
print("Softmax:", np.round(softmax(np.array([2.0, 1.0, 0.1])), 3))

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x = torch.tensor(2.0)
t = torch.tensor(10.0)

loss = (w * x + b - t) ** 2
loss.backward()

print("Weight gradient:", w.grad.item())
print("Bias gradient:", b.grad.item())

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)
np.random.seed(42)

W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.5
epochs = 5000
loss_history = []

for epoch in range(epochs):

    hidden = relu(np.dot(X, W1) + b1)
    predictions = sigmoid(np.dot(hidden, W2) + b2)

    loss = -np.mean(
        y * np.log(predictions + 1e-8)
        + (1 - y) * np.log(1 - predictions + 1e-8)
    )
    loss_history.append(loss)

    dz2 = (predictions - y) / len(X)
    dW2 = np.dot(hidden.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dhidden = np.dot(dz2, W2.T)
    dz1 = dhidden * ((np.dot(X, W1) + b1) > 0)

    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

hidden = relu(np.dot(X, W1) + b1)
predictions = sigmoid(np.dot(hidden, W2) + b2)

print("\nXOR Predictions:",np.round(predictions.flatten(), 2))

print("\nPredicted Classes:",(predictions.flatten() >= 0.5).astype(int))

print("\nFinal Loss:", round(loss_history[-1], 4))

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("XOR Neural Network Training Loss")
plt.grid(True)
plt.savefig("task1/xor_loss.png")
plt.show()

def train_xor(lr, epochs=5000):
    np.random.seed(42)

    w1 = np.random.randn(2, 4) * np.sqrt(2 / 2)
    b1 = np.zeros((1, 4))
    w2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):
        z1 = np.dot(X, w1) + b1
        h = relu(z1)
        p = sigmoid(np.dot(h, w2) + b2)

        loss = -np.mean(
            y * np.log(p + 1e-8)
            + (1 - y) * np.log(1 - p + 1e-8)
        )
        losses.append(loss)

        dz2 = (p - y) / len(X)
        dw2 = h.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh = dz2 @ w2.T
        dz1 = dh * (z1 > 0)
        dw1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

        if not np.isfinite(loss) or loss > 1e6:
            break

    return losses

plt.figure(figsize=(8, 5))

for lr in [0.01, 0.5, 5]:
    losses = train_xor(lr)
    plt.plot(losses, label=f"Learning rate = {lr}")
    print(f"LR={lr}, final recorded loss={losses[-1]:.4f}")

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("task1/learning_rate_comparison.png")
plt.show()