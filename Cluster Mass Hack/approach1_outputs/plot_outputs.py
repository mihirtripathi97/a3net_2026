from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
metrics = pd.read_csv(OUTPUT_DIR / 'baseline_metrics_by_epoch.csv')

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for axis, metric in zip(axes, ['r2', 'mae', 'rmse', 'val_loss']):
    axis.plot(metrics['epochs'], metrics[metric], 'o-', color='C0')
    axis.set(title=metric, xlabel='Training epochs')
    axis.grid(alpha=0.25)
fig.suptitle('Baseline performance versus training epochs')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'baseline_performance_vs_epochs.png', dpi=200, bbox_inches='tight')
plt.close(fig)

for epoch_count in metrics['epochs']:
    history = np.load(OUTPUT_DIR / f'training_history_epoch_{int(epoch_count)}.npz')
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(history['loss'], label='Training loss')
    axis.plot(history['val_loss'], label='Validation loss')
    axis.set(xlabel='Epoch', ylabel='MSE', title=f'Loss curves: {int(epoch_count)} epochs')
    axis.set_yscale('log')
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'loss_curves_epoch_{int(epoch_count)}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
print(f'Plots regenerated in {OUTPUT_DIR}')
