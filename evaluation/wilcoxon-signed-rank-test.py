import itertools
from typing import List
import scipy.stats as stats

def run_wilcoxon_tests(
            metric_data: List[List[float]],
            alpha: float = 0.05,
            target: str = "alpha_FF5MOM",
            zero_method: str = "wilcox"
        ):
    """
    Runs Wilcoxon signed-rank tests for pairwise comparisons of model evaluation metrics.

    Args:
        metric_data (List[List[float]]): A list of lists, where each sublist contains evaluation metrics for a model.
        alpha (float): Significance level for hypothesis testing.
        target (str): Name of the target variable used in the model.
        zero_method (str): Method for handling zero-differences in Wilcoxon test. Options: 'wilcox', 'pratt', or 'zsplit'.
    """
    print(f"\n===== Wilcoxon Signed-Rank Test: {target} as Target =====")
    total_models = len(metric_data)
    comparisons = itertools.combinations(range(total_models), 2)

    for i, j in comparisons:
        model_1 = metric_data[i]
        model_2 = metric_data[j]
        try:
            statistic, p_value = stats.wilcoxon(model_1, model_2, zero_method=zero_method)
        except ValueError as e:
            print(f"Comparison {i+1} vs {j+1} failed: {e}")
            continue

        print(f"\nComparing Model {i+1} and Model {j+1}")
        print(f"Statistic: {statistic:.4f}")
        print(f"P-Value: {p_value:.4f}")

        if p_value < alpha:
            print("✅ Statistically significant difference in predictive power.")
        else:
            print("❌ No statistically significant difference found.")
        print("------------------------------------------------")

# Test 1: alpha_FF5MOM as target
metrics_alpha = [
    [1.70129339425417E-06, 0.777226228626114, 0.000107728795889211],  # Gradient Boosting
    [1.79947722079540E-06, 0.287829699824572, 0.000122642680432675],  # Random Forest Grid Search
    [1.79947722079540E-06, 0.287829699824573, 0.000122642680432675],  # Random Forest Random Search
]

run_wilcoxon_tests(metrics_alpha, target="alpha_FF5MOM", zero_method="zsplit")

# Test 2: t_alpha_FF5MOM as target
metrics_t_alpha = [
    [7.63710359699315E-03, 0.152183911071124, 0.053821072028438900],  # Gradient Boosting
    [1.32157968208173E-02, 0.155805293699349, 0.058738450136528100],  # Random Forest Grid Search
    [1.32157968208173E-02, 0.155805293699349, 0.058738450136528100],  # Random Forest Random Search
]

run_wilcoxon_tests(metrics_t_alpha, target="t_alpha_FF5MOM", zero_method="wilcox")