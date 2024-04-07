#Import packages to run the tests
import itertools
import scipy.stats as stats

#alpha_FF5MOM as target variable

#Create array of evaluation metrics for three models
model_criteria = [
    [1.70129339425417E-06, 0.777226228626114, 0.000107728795889211], #gradient_boosting
    [1.79947722079540E-06, 0.287829699824572, 0.000122642680432675], #random_forest_grid_search_cv
    [1.79947722079540E-06, 0.287829699824573, 0.000122642680432675] #random_forest_rand_search_cv
]

#Perform the pairwise comparison using Wilcoxon signed-rank test
alpha = 0.05
total_models = len(model_criteria)
comp = list(itertools.combinations(range(total_models), 2))

#Create for loop
for i, j in comp:
    model_1 = model_criteria[i]
    model_2 = model_criteria[j]
    
    statistic, p_value = stats.wilcoxon(model_1, model_2, zero_method="zsplit")


    #Print results
    print(f"Comparing Model {i+1} and Model {j+1}")
    print("Wilcoxon Signed-Rank Test Results")
    print("Statistic:", statistic)
    print("P-Value:", p_value)

    #Interpret results
    if p_value < alpha:
        print('statistically significant difference in predictive power between at least two models')
    else:
        print('there is no statistically significant difference in predictive power between the models')

    #Add separation line for clarity
    print("-----------------------------------------------")
    
    
    
#---------------------------------------------------------------------------------------------------#
    
#t_alpha_FF5MOM as target variable

#Import packages to run the tests
import itertools
import scipy.stats as stats

#Create array of evaluation metrics for three models
model_criteria = [
    [7.63710359699315E-03, 0.152183911071124, 0.053821072028438900], #gradient_boosting
    [1.32157968208173E-02, 0.155805293699349, 0.058738450136528100], #random_forest_grid_search_cv
    [1.32157968208173E-02, 0.155805293699349, 0.058738450136528100] #random_forest_rand_search_cv
]

#Perform the pairwise comparison using Wilcoxon signed-rank test
alpha = 0.05
total_models = len(model_criteria)
comp = list(itertools.combinations(range(total_models), 2))

#Create for loop
for i, j in comp:
    model_1 = model_criteria[i]
    model_2 = model_criteria[j]
    
    statistic, p_value = stats.wilcoxon(model_1, model_2, zero_method="wilcox")


    #Print results
    print(f"Comparing Model {i+1} and Model {j+1}")
    print("Wilcoxon Signed-Rank Test Results")
    print("Statistic:", statistic)
    print("P-Value:", p_value)

    #Interpret results
    if p_value < alpha:
        print('statistically significant difference in predictive power between at least two models')
    else:
        print('there is no statistically significant difference in predictive power between the models')

    #Add separation line for clarity
    print("-----------------------------------------------")
    
    
    
#---------------------------------------------------------------------------------------------------#
