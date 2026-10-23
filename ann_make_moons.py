"""
Project 1: Artificial Neural Network for binary classification on make_moons.
Architecture: 2 inputs -> 16 hidden (tanh) -> 8 hidden (tanh) -> 1 output (sigmoid)
Implemented from scratch with NumPy (forward pass, backprop, gradient descent).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ---------- 1. Data ----------
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 2. Network ----------
class ANN:
    def __init__(self, layers=(2, 16, 8, 1), lr=0.1):
        self.lr = lr
        self.W, self.b = [], []
        for i in range(len(layers) - 1):
            # Xavier initialisation
            self.W.append(np.random.randn(layers[i], layers[i + 1]) * np.sqrt(1 / layers[i]))
            self.b.append(np.zeros((1, layers[i + 1])))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.A = [X]
        for i in range(len(self.W) - 1):
            self.A.append(np.tanh(self.A[-1] @ self.W[i] + self.b[i]))
        self.A.append(self.sigmoid(self.A[-1] @ self.W[-1] + self.b[-1]))
        return self.A[-1]

    def loss(self, y, y_hat, eps=1e-8):
        # Binary cross-entropy
        return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    def backward(self, y):
        m = y.shape[0]
        dZ = self.A[-1] - y  # sigmoid + BCE simplification
        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dZ / m
            db = dZ.mean(axis=0, keepdims=True)
            if i > 0:
                dZ = (dZ @ self.W[i].T) * (1 - self.A[i] ** 2)  # tanh derivative
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def fit(self, X, y, X_val, y_val, epochs=2000):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        for epoch in range(epochs):
            y_hat = self.forward(X)
            self.backward(y)
            if epoch % 10 == 0:
                history["train_loss"].append(self.loss(y, self.forward(X)))
                history["val_loss"].append(self.loss(y_val, self.forward(X_val)))
                history["train_acc"].append((self.predict(X) == y).mean())
                history["val_acc"].append((self.predict(X_val) == y_val).mean())
            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d} | loss {history['train_loss'][-1]:.4f} "
                      f"| train acc {history['train_acc'][-1]:.3f} "
                      f"| val acc {history['val_acc'][-1]:.3f}")
        return history

# ---------- 3. Train ----------
model = ANN()
history = model.fit(X_train, y_train, X_test, y_test, epochs=2000)

train_acc = (model.predict(X_train) == y_train).mean()
test_acc = (model.predict(X_test) == y_test).mean()
print(f"\nFinal train accuracy: {train_acc:.3f}")
print(f"Final test accuracy:  {test_acc:.3f}")

# ---------- 4. Plots ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

ax[0].plot(history["train_loss"], label="train")
ax[0].plot(history["val_loss"], label="test")
ax[0].set_title("Loss (binary cross-entropy)")
ax[0].set_xlabel("Every 10 epochs")
ax[0].legend()

ax[1].plot(history["train_acc"], label="train")
ax[1].plot(history["val_acc"], label="test")
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Every 10 epochs")
ax[1].legend()

xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax[2].contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
ax[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test.ravel(), cmap="RdBu", edgecolors="k", s=20)
ax[2].set_title(f"Decision boundary (test acc {test_acc:.1%})")

plt.tight_layout()
plt.savefig("ann_make_moons_results.png", dpi=120)
print("Saved ann_make_moons_results.png")
