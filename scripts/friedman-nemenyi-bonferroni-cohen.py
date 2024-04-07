#alpha_FF5MOM target variable

#Import packages to run the tests
import scipy.stats as stats

#Create array of evaluation metrics for three models
gradient_boosting = [1.70129339425417E-06, 0.777226228626114, 0.000107728795889211]
random_forest_grid_search_cv = [1.79947722079540E-06, 0.287829699824572, 0.000122642680432675]
random_forest_rand_search_cv = [1.79947722079540E-06, 0.287829699824573, 0.000122642680432675]

#Execute the Friedman test
f_value, p_value = stats.friedmanchisquare(gradient_boosting, random_forest_grid_search_cv, random_forest_rand_search_cv)

#Set alpha
alpha = 0.05

#Interpret results
print("Friedman Test Results")
print("Statistic", f_value)
print("P-Value", p_value)

#Interpret results
if p_value < alpha:
    print('statistically significant difference in predictive power between at least two models')
else:
    print('there is no statistically significant difference in predictive power between the models')
    

#------------------------------------------------------------------------------------------------------------------#    

#t_alphaFF5MOM target variable

#Import packages to run the tests
import scipy.stats as stats

#Create array of evaluation metrics for three models
gradient_boosting = [7.63710359699315E-03, 0.152183911071124, 0.053821072028438900]
random_forest_grid_search_cv = [1.32157968208173E-02, 0.155805293699349, 0.058738450136528100]
random_forest_rand_search_cv = [1.32157968208173E-02, 0.155805293699349, 0.058738450136528100]

#Execute the Friedman test
f_value, p_value = stats.friedmanchisquare(gradient_boosting, random_forest_grid_search_cv, random_forest_rand_search_cv)

#Set alpha
alpha = 0.05

#Print results
print("Friedman Test Results")
print("Statistic", f_value)
print("P-Value", p_value)

#Interpret results
if p_value < alpha:
    print('statistically significant difference in predictive power between at least two models')
else:
    print('there is no statistically significant difference in predictive power between the models')
    
#Says there is a difference between these two tests

#Import Packages to run ad-hoc tests
import numpy as np
import pandas as pd
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi
from statsmodels.stats.multitest import multipletests

#Create a dataframe of evaluation metrics for the models
model_metrics = pd.DataFrame({
    'Model': ['gradient_boosting', 'random_forest_grid_search_cv', 'random_forest_rand_search_cv'],
    'MSE': [7.63710359699315E-03, 1.32157968208173E-02, 1.32157968208173E-02],
    'MAPE': [0.152183911071124, 0.155805293699349, 0.155805293699349],
    'MAE': [0.053821072028438900, 0.058738450136528100, 0.058738450136528100]
    })

#Convert df to NumPy array
data = model_metrics.iloc[:, 1:].to_numpy()

#Set alpha
alpha = 0.05

#Perform the Nemenyi post-hoc test
posthoc_results = posthoc_nemenyi(data)

#Print results
print("Nemenyi Post-Hoc Test Results:")
print(posthoc_results)

#Perform the Bonferroni-Dunn test
num_models = len(model_metrics)
adjusted_alpha = alpha / (num_models * (num_models - 1) / 2)

#Create an array to store the test statistics and p-values
test_stats = np.zeros((num_models, num_models))
p_values = np.zeros((num_models, num_models))

#Perform pairwise comparisons and calculate test statistics and p-values
for i in range(num_models):
    for j in range(i + 1, num_models):
        test_stat, p_val = stats.ranksums(data[:, i], data[:, j])
        test_stats[i, j] = test_stat
        p_values[i, j] = p_val

#Adjust the p-values using the Bonferroni correction
reject, adjusted_p_values, _, _ = multipletests(p_values[np.triu_indices(num_models, k=1)], alpha=adjusted_alpha)

#Print the adjusted p-values for each model comparison
print("Bonferroni-Dunn Test Results:")
index = 0
for i in range(num_models):
    for j in range(i + 1, num_models):
        print(f"Comparison: Model {i+1} vs Model {j+1}")
        print("Test Statistic:", test_stats[i, j])
        print("Adjusted p-value:", adjusted_p_values[index])
        print("-----------------------------------------------")
        index += 1

#Perform Effect Size Test to Evaluate Practical Significance - Cohen's d

#Calculate MSE effect size gradient_boosting vs. random_forest models
mse_gradient_boosting = 7.63710359699315E-03
mse_random_forest_models = 1.32157968208173E-02
pooled_std_dev = np.sqrt((mse_gradient_boosting + mse_random_forest_models) / 2)
effect_size_gradient_boosting_vs_random_forest_models = (mse_gradient_boosting - mse_random_forest_models) / pooled_std_dev

#Print results
print("MSE Effect Size:")
print(abs(effect_size_gradient_boosting_vs_random_forest_models))

#Check for significance
if abs(effect_size_gradient_boosting_vs_random_forest_models) >= 0.2:
    print("There is a small practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models) >= 0.5:
    print("There is a medium practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models) >= 0.8:
    print("There is a large practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models) <= 0.2:
    print("There is no practical effect size")

#Insert line break
print('-----------------------------------------')

#Calculate MAPE effect size gradient_boosting vs. random_forest models
mape_gradient_boosting = 0.152183911071124
mape_random_forest_models = 0.155805293699349
pooled_std_dev = np.sqrt((mape_gradient_boosting + mape_random_forest_models) / 2)
effect_size_gradient_boosting_vs_random_forest_models2 = (mape_gradient_boosting - mape_random_forest_models) / pooled_std_dev

#Print results
print("MAPE Effect Size:")
print(abs(effect_size_gradient_boosting_vs_random_forest_models2))

#Check for significance
if abs(effect_size_gradient_boosting_vs_random_forest_models2) >= 0.2:
    print("There is a small practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models2) >= 0.5:
    print("There is a medium practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models2) >= 0.8:
    print("There is a large practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models2) <= 0.2:
    print("There is no practical effect size")

#Insert line break
print('-----------------------------------------')

#Calculate MAE effect size gradient_boosting vs. random_forest models
mae_gradient_boosting = 0.053821072028438900
mae_forest_models = 0.058738450136528100
pooled_std_dev = np.sqrt((mae_gradient_boosting + mae_forest_models) / 2)
effect_size_gradient_boosting_vs_random_forest_models3 = ((mae_gradient_boosting - mae_forest_models) / 2)

#Print results
print("MAE Effect Size:")
print(abs(effect_size_gradient_boosting_vs_random_forest_models3))

#Check for significance
if abs(effect_size_gradient_boosting_vs_random_forest_models3) >= 0.2:
    print("There is a small practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models3) >= 0.5:
    print("There is a medium practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models3) >= 0.8:
    print("There is a large practical difference between gradient_boosting and random_forest models")
if abs(effect_size_gradient_boosting_vs_random_forest_models3) <= 0.2:
    print("There is no practical effect size")
