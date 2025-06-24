import numpy as np
import pandas as pd
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi
from statsmodels.stats.multitest import multipletests

# ------------------------- Friedman Tests ------------------------- #

def run_friedman_test(metric_name: str, *models: list[float], alpha: float = 0.05) -> None:
    """Run Friedman test on multiple models' performance metrics."""
    print(f"\n===== Friedman Test: {metric_name} =====")
    f_value, p_value = stats.friedmanchisquare(*models)
    print("Statistic:", f_value)
    print("P-Value:", p_value)
    if p_value < alpha:
        print("Statistically significant difference in predictive power between at least two models.")
    else:
        print("No statistically significant difference in predictive power between the models.")

# Run for alpha_FF5MOM
run_friedman_test("alpha_FF5MOM", 
    [1.701e-6, 0.7772, 1.077e-4],
    [1.799e-6, 0.2878, 1.226e-4],
    [1.799e-6, 0.2878, 1.226e-4]
)

# Run for t_alphaFF5MOM
run_friedman_test("t_alphaFF5MOM", 
    [7.637e-3, 0.1522, 0.0538],
    [1.322e-2, 0.1558, 0.0587],
    [1.322e-2, 0.1558, 0.0587]
)

# ---------------------- Post-hoc & Effect Size ---------------------- #

model_names = ['gradient_boosting', 'random_forest_grid_cv', 'random_forest_rand_cv']
model_metrics = pd.DataFrame({
    'Model': model_names,
    'MSE': [7.637e-3, 1.322e-2, 1.322e-2],
    'MAPE': [0.1522, 0.1558, 0.1558],
    'MAE': [0.0538, 0.0587, 0.0587]
})

# Nemenyi Post-hoc Test
print("\n===== Nemenyi Post-Hoc Test =====")
data = model_metrics.drop(columns="Model").to_numpy()
posthoc_results = posthoc_nemenyi(data)
print(posthoc_results)

# Bonferroni-Dunn Test
print("\n===== Bonferroni-Dunn Test =====")
alpha = 0.05
n_models = len(model_names)
adjusted_alpha = alpha / (n_models * (n_models - 1) / 2)
test_stats = np.zeros((n_models, n_models))
p_values = np.zeros((n_models, n_models))

for i in range(n_models):
    for j in range(i + 1, n_models):
        stat, p_val = stats.ranksums(data[:, i], data[:, j])
        test_stats[i, j] = stat
        p_values[i, j] = p_val

reject, adjusted_p, _, _ = multipletests(p_values[np.triu_indices(n_models, k=1)], alpha=adjusted_alpha)

idx = 0
for i in range(n_models):
    for j in range(i + 1, n_models):
        print(f"Comparison: {model_names[i]} vs {model_names[j]}")
        print("Test Statistic:", test_stats[i, j])
        print("Adjusted P-Value:", adjusted_p[idx])
        print("-----------------------------------")
        idx += 1

# ---------------------- Effect Size (Cohen's d) ---------------------- #

def compute_effect_size(metric_name: str, val1: float, val2: float) -> None:
    """Computes and interprets Cohen's d effect size between two values."""
    pooled_std = np.sqrt((val1 + val2) / 2)
    effect_size = (val1 - val2) / pooled_std
    abs_effect = abs(effect_size)

    print(f"{metric_name} Effect Size:")
    print(f"Cohen's d: {abs_effect:.4f}")

    if abs_effect >= 0.8:
        print("Large practical difference")
    elif abs_effect >= 0.5:
        print("Medium practical difference")
    elif abs_effect >= 0.2:
        print("Small practical difference")
    else:
        print("No practical difference")
    print("-----------------------------------")

compute_effect_size("MSE", 7.637e-3, 1.322e-2)
compute_effect_size("MAPE", 0.1522, 0.1558)
compute_effect_size("MAE", 0.0538, 0.0587)
