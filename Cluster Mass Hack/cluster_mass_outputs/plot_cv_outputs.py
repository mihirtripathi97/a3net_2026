from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent


def save_cv_plots(result_table, histories, prefix):
    metrics = ['r2', 'mae', 'rmse', 'scatter', 'bias', 'val_loss']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, metric in zip(axes.flat, metrics):
        axis.bar(result_table['fold'], result_table[metric], color='C2')
        axis.set_title(metric)
        axis.set_xlabel('Fold')
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle(f'{prefix}: validation metrics by fold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{prefix}_metrics_by_fold.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    train_loss = histories[f'{prefix}_train_loss']
    val_loss = histories[f'{prefix}_val_loss']
    epoch_axis = np.arange(1, train_loss.shape[1] + 1)
    fig, axis = plt.subplots(figsize=(10, 6))
    for losses, label, color in [
        (train_loss, 'Training', 'C0'),
        (val_loss, 'Validation', 'C1')
    ]:
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)
        axis.plot(epoch_axis, mean_loss, color=color, label=f'{label} mean')
        axis.fill_between(
            epoch_axis,
            mean_loss - std_loss,
            mean_loss + std_loss,
            color=color,
            alpha=0.18,
            label=f'{label} +/- 1 std'
        )
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Mean squared error')
    axis.set_title(f'{prefix}: mean loss and fold-to-fold variation')
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{prefix}_loss_curves.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


for prefix in ['no_rotation', 'with_rotation']:
    results = pd.read_csv(OUTPUT_DIR / f'cv_{prefix}.csv')
    save_cv_plots(results, np.load(OUTPUT_DIR / 'training_histories.npz'), prefix)
print(f'Plots regenerated in {OUTPUT_DIR}')
