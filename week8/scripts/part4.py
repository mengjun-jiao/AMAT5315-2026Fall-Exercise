"""Part 4: recall the supplied 32x32 算 bitmap from random key-value memories."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root = Path(__file__).resolve().parent.parent
bitmap_path = root / 'inputs' / 'suan_32.png'
with Image.open(bitmap_path) as image:
    image = np.asarray(image.convert('L'))
assert image.shape == (32, 32) and set(np.unique(image)).issubset({0, 255})
picture = np.where(image.ravel() > 0, 1., -1.)
rng = np.random.default_rng(2026)
fig, axs = plt.subplots(4, 4, figsize=(11, 10.5), constrained_layout=True)
store_names = ['Outer product', 'MLP, ReLU', 'MLP, ReLU²', 'Attention, softmax β=8']
results, recalls = {}, {}
yy, xx = np.indices((32, 32))
for row, M in enumerate([64, 256, 1024, 4096]):
    keys = rng.normal(size=(M, 64))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.choice([-1., 1.], size=(M, 1024))
    values[0] = picture
    similarities = keys @ keys[0]
    similarities[0] = 1.
    exponential = np.exp(8 * (similarities - similarities.max()))
    weights = [similarities, np.maximum(similarities, 0),
               np.maximum(similarities, 0)**2, exponential / exponential.sum()]
    for col, (name, w) in enumerate(zip(['outer', 'relu', 'relu2', 'softmax'], weights)):
        own_weight = w[0]
        assert own_weight > 0
        normalized = w @ values / own_weight
        interference = w[1:] @ values[1:] / own_weight
        np.testing.assert_allclose(normalized, picture + interference, atol=1e-12)
        ratio = float(np.linalg.norm(interference) / np.linalg.norm(picture))
        results[f'{M}_{name}'] = ratio
        recalls[f'{M}_{name}'] = normalized.reshape(32, 32)
        ax = axs[row, col]
        # Pixel value -1 corresponds to no ink; dot area is clipped only for display.
        dot_area = np.clip((normalized + 1) * 2, 0, 12)
        ax.scatter(xx.ravel(), -yy.ravel(), s=dot_area,
                   color=('#c6372b' if ratio < 1 else '#555555'), linewidths=0)
        ax.set(aspect='equal', xticks=[], yticks=[])
        ax.set_title(f'R = {ratio:.3f}', fontsize=10, pad=3)
        if row == 0:
            ax.text(.5, 1.15, store_names[col], ha='center', va='bottom',
                    transform=ax.transAxes, fontsize=10)
        if col == 0:
            ax.set_ylabel(f'M = {M}', fontsize=10)
fig.suptitle('Recalling a 32 × 32 character among random memories', fontsize=14)
fig.savefig(root / 'evidence' / 'part4_recall.png', dpi=180)
np.savez_compressed(root / 'artifacts' / 'part4_recalls.npz', target=picture.reshape(32, 32), **recalls)
(root / 'artifacts' / 'part4.json').write_text(json.dumps({
    'seed': 2026, 'target_character': '算', 'target_bitmap': 'inputs/suan_32.png',
    'dimension': 64, 'sharpness': 8, 'ratios': results,
}, ensure_ascii=False, indent=2), encoding='utf8')
assert all(results[f'{m}_softmax'] < .1 for m in [64, 256, 1024, 4096])
assert results['4096_relu2'] > 1
assert abs(results['1024_outer'] / 4 - 1) < .15
assert abs(results['1024_relu'] / np.sqrt(1024 / 128) - 1) < .15
print(json.dumps(results, indent=2))
