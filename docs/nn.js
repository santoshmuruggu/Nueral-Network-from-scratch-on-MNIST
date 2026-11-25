// Port of draw_predict.py: same brush, same MNIST-style preprocessing,
// same 784 -> 128 -> 10 forward pass, using the weights from model.pkl.
(function (root) {
  const CANVAS_SIZE = 280;
  const BRUSH_RADIUS = 8;
  const N_IN = 784, N_HID = 128, N_OUT = 10;

  // Python's round() rounds half to even; Math.round doesn't.
  function roundHalfEven(x) {
    const f = Math.floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }

  // Same disc mask as App.paint().
  function stamp(grid, x, y, r = BRUSH_RADIUS, size = CANVAS_SIZE) {
    const y0 = Math.max(0, y - r), y1 = Math.min(size, y + r);
    const x0 = Math.max(0, x - r), x1 = Math.min(size, x + r);
    for (let yy = y0; yy < y1; yy++) {
      for (let xx = x0; xx < x1; xx++) {
        if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) grid[yy * size + xx] = 1;
      }
    }
  }

  function resizeNearest(img, inH, inW, outH, outW) {
    const out = new Float32Array(outH * outW);
    for (let i = 0; i < outH; i++) {
      const r = Math.min(Math.trunc((i * inH) / outH), inH - 1);
      for (let j = 0; j < outW; j++) {
        const c = Math.min(Math.trunc((j * inW) / outW), inW - 1);
        out[i * outW + j] = img[r * inW + c];
      }
    }
    return out;
  }

  // Crop to the ink, fit the longer side to 20px, center in 28x28.
  function toMnistFormat(grid, size = CANVAS_SIZE, threshold = 0.3) {
    let y0 = size, y1 = -1, x0 = size, x1 = -1;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        if (grid[y * size + x] > threshold) {
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
        }
      }
    }
    if (y1 < 0) return null;
    y1 += 1; x1 += 1;

    const h = y1 - y0, w = x1 - x0;
    const cropped = new Float32Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) cropped[y * w + x] = grid[(y0 + y) * size + (x0 + x)];
    }

    let newH, newW;
    if (h > w) { newH = 20; newW = Math.max(1, roundHalfEven((w * 20) / h)); }
    else { newW = 20; newH = Math.max(1, roundHalfEven((h * 20) / w)); }
    const resized = resizeNearest(cropped, h, w, newH, newW);

    const out = new Float32Array(N_IN);
    const top = Math.floor((28 - newH) / 2), left = Math.floor((28 - newW) / 2);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) out[(top + y) * 28 + left + x] = resized[y * newW + x];
    }
    return out;
  }

  // weights.bin = W1 (784x128), b1 (128), W2 (128x10), b2 (10), float32 little-endian, row-major.
  function loadModel(buffer) {
    const all = new Float32Array(buffer);
    let o = 0;
    const take = (n) => { const a = all.subarray(o, o + n); o += n; return a; };
    const model = { W1: take(N_IN * N_HID), b1: take(N_HID), W2: take(N_HID * N_OUT), b2: take(N_OUT) };
    if (o !== all.length) throw new Error("unexpected weights size");
    return model;
  }

  function forward(m, x) {
    const a1 = new Float64Array(N_HID);
    for (let j = 0; j < N_HID; j++) a1[j] = m.b1[j];
    for (let i = 0; i < N_IN; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * N_HID;
      for (let j = 0; j < N_HID; j++) a1[j] += xi * m.W1[row + j];
    }
    for (let j = 0; j < N_HID; j++) if (a1[j] < 0) a1[j] = 0;

    const z2 = new Float64Array(N_OUT);
    for (let k = 0; k < N_OUT; k++) z2[k] = m.b2[k];
    for (let j = 0; j < N_HID; j++) {
      const aj = a1[j];
      if (aj === 0) continue;
      const row = j * N_OUT;
      for (let k = 0; k < N_OUT; k++) z2[k] += aj * m.W2[row + k];
    }

    let max = -Infinity, sum = 0;
    for (let k = 0; k < N_OUT; k++) if (z2[k] > max) max = z2[k];
    for (let k = 0; k < N_OUT; k++) { z2[k] = Math.exp(z2[k] - max); sum += z2[k]; }
    for (let k = 0; k < N_OUT; k++) z2[k] /= sum;
    return z2;
  }

  const api = { CANVAS_SIZE, BRUSH_RADIUS, stamp, toMnistFormat, loadModel, forward };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NN = api;
})(this);
