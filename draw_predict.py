import pickle
import numpy as np
import tkinter as tk

CANVAS_SIZE = 280   # 10x the 28x28 MNIST resolution, easier to draw on
SCALE = CANVAS_SIZE // 28
BRUSH_RADIUS = 8

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X):
    a1 = relu(X @ W1 + b1)
    a2 = softmax(a1 @ W2 + b2)
    return a2


def resize_nearest(img, out_h, out_w):
    in_h, in_w = img.shape
    row_idx = np.minimum((np.arange(out_h) * in_h / out_h).astype(int), in_h - 1)
    col_idx = np.minimum((np.arange(out_w) * in_w / out_w).astype(int), in_w - 1)
    return img[row_idx][:, col_idx]


def to_mnist_format(grid, threshold=0.3):
    """Crop to the drawn strokes, scale to fit a 20x20 box, and center in
    a 28x28 frame -- mirrors how MNIST digits were originally preprocessed."""
    ys, xs = np.nonzero(grid > threshold)
    if ys.size == 0:
        return None

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = grid[y0:y1, x0:x1]

    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, round(w * 20 / h))
    else:
        new_w = 20
        new_h = max(1, round(h * 20 / w))
    resized = resize_nearest(cropped, new_h, new_w)

    out = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


class App:
    def __init__(self, root):
        self.root = root
        root.title("Draw a digit")

        # pixel grid mirroring the canvas, black background (0) / white strokes (1)
        self.grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.result_label = tk.Label(root, text="Draw a digit, then click Predict", font=("Helvetica", 16))
        self.result_label.grid(row=1, column=0, columnspan=2, pady=8)

        predict_btn = tk.Button(root, text="Predict", command=self.predict, font=("Helvetica", 12))
        predict_btn.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        clear_btn = tk.Button(root, text="Clear", command=self.clear, font=("Helvetica", 12))
        clear_btn.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

    def paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")

        y0, y1 = max(0, y - r), min(CANVAS_SIZE, y + r)
        x0, x1 = max(0, x - r), min(CANVAS_SIZE, x + r)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        self.grid[y0:y1, x0:x1][mask] = 1.0

    def clear(self):
        self.canvas.delete("all")
        self.grid[:] = 0.0
        self.result_label.config(text="Draw a digit, then click Predict")

    def predict(self):
        small = to_mnist_format(self.grid)
        if small is None:
            self.result_label.config(text="Draw something first!")
            return

        x = small.reshape(1, 784)
        probs = forward(x)[0]
        digit = int(np.argmax(probs))
        confidence = probs[digit] * 100

        self.result_label.config(text=f"Prediction: {digit}  ({confidence:.1f}% confident)")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
