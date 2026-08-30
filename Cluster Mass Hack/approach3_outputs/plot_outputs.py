from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent
results = pd.read_csv(OUTPUT_DIR / 'cv_augmented.csv')
predictions = np.load(OUTPUT_DIR / 'fold_predictions_augmented.npz', allow_pickle=True)
metadata = __import__('json').loads((OUTPUT_DIR / 'run_metadata.json').read_text())
norm_cv = metadata.get('normalization_offset', 0.0)
true_by_fold = predictions['true']
predicted_by_fold = predictions['predicted']
all_true = np.concatenate(true_by_fold) + norm_cv
all_predicted = np.concatenate(predicted_by_fold) + norm_cv
best_fold = int(results.loc[results['r2'].idxmax(), 'fold'])
best_true = true_by_fold[best_fold - 1] + norm_cv
best_predicted = predicted_by_fold[best_fold - 1] + norm_cv

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for axis, metric in zip(axes.flat, ['r2', 'mae', 'rmse', 'scatter', 'bias', 'slope_true_predicted']):
    axis.bar(results['fold'], results[metric], color='C3')
    axis.set(title=metric, xlabel='Fold')
    axis.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'augmented_cv_metrics_by_fold.png', dpi=200, bbox_inches='tight')
plt.close(fig)

fig, axis = plt.subplots(figsize=(10, 6))
for values, label, color in [(all_predicted - all_true, '10-fold mean', 'C0'), (best_predicted - best_true, f'Best fit fold ({best_fold})', 'C3')]:
    axis.hist(values, bins=30, density=True, alpha=0.28, color=color, label=f'{label} PDF')
    points = np.linspace(values.min(), values.max(), 300)
    mean, sigma = np.mean(values), np.std(values, ddof=1)
    axis.plot(points, np.exp(-0.5 * ((points - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), color=color, label=f'{label}: mean={mean:.3f}, sigma={sigma:.3f}')
axis.axvline(0, color='black', linestyle='--')
axis.set(xlabel='log(Mpredicted) - log(Mtrue)', ylabel='Probability density', title='Mass-error PDFs with Gaussian fits')
axis.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'mass_error_pdfs_gaussian_fit.png', dpi=200, bbox_inches='tight')
plt.close(fig)

limits = [min(all_true.min(), all_predicted.min()), max(all_true.max(), all_predicted.max())]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for axis, true_values, predicted_values, title in zip(axes, [all_true, best_true], [all_predicted, best_predicted], ['10-fold mean', f'Best fit fold ({best_fold})']):
    axis.scatter(true_values, predicted_values, s=12, alpha=0.35)
    axis.plot(limits, limits, 'k--', linewidth=1.5, label='1:1')
    axis.set(title=title, xlabel='True log(M500)', xlim=limits, ylim=limits)
    axis.grid(alpha=0.25)
    axis.legend()
axes[0].set_ylabel('Predicted log(M500)')
fig.suptitle('Predicted mass as a function of true mass')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'predicted_vs_true_mass.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'Plots regenerated in {OUTPUT_DIR}')
