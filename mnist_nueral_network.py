import pickle 
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f: 
    train, val, test = pickle.load(f,encoding='latin1')

X_train, y_train = train
print(X_train.shape)
print(y_train.shape) 

print(y_train[0])
print(X_train[0].shape)
print(X_train[0].min(), X_train[0].max())


import numpy as np 
def one_hot(y, num_classes=10): 
    onehot = np.zeros((y.shape[0], num_classes))
    onehot[np.arange(y.shape[0]), y] = 1 
    return onehot

y_train_onehot = one_hot(y_train)
print(y_train_onehot[0])
print(y_train_onehot.shape)


# ---- network setup ----
input_size = X_train.shape[1]   # 784
hidden_size = 128
output_size = 10

np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros(output_size)


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(z.dtype)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def cross_entropy_loss(probs, y_onehot):
    n = probs.shape[0]
    correct_logprobs = -np.log(probs[np.arange(n), y_onehot.argmax(axis=1)] + 1e-9)
    return np.sum(correct_logprobs) / n


def backward(X, y_onehot, z1, a1, a2):
    n = X.shape[0]

    dz2 = (a2 - y_onehot) / n           # (n, output_size)
    dW2 = a1.T @ dz2                    # (hidden_size, output_size)
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T                    # (n, hidden_size)
    dz1 = da1 * relu_deriv(z1)          # (n, hidden_size)
    dW1 = X.T @ dz1                     # (input_size, hidden_size)
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2


def accuracy(X, y):
    _, _, _, probs = forward(X)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)


# ---- training loop ----
learning_rate = 0.5
epochs = 10
batch_size = 64
n_samples = X_train.shape[0]

X_val, y_val = val
X_test, y_test = test

for epoch in range(epochs):
    perm = np.random.permutation(n_samples)
    X_shuffled = X_train[perm]
    y_shuffled = y_train_onehot[perm]

    epoch_loss = 0.0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        z1, a1, z2, a2 = forward(X_batch)
        loss = cross_entropy_loss(a2, y_batch)
        dW1, db1, dW2, db2 = backward(X_batch, y_batch, z1, a1, a2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        epoch_loss += loss
        n_batches += 1

    val_acc = accuracy(X_val, y_val)
    print(f"epoch {epoch+1}/{epochs}  loss: {epoch_loss/n_batches:.4f}  val_acc: {val_acc:.4f}")

test_acc = accuracy(X_test, y_test)
print(f"test accuracy: {test_acc:.4f}")

# ---- save the trained model ----
model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("model saved to model.pkl")

