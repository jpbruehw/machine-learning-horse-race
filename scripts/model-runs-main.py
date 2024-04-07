#import .txt and packages to python
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("/Users/yannickarfaoui/dataset_module_10.txt", delimiter=" ", low_memory=False)

#Drop blanks from FUND_NAME
df = df.dropna(axis=0, subset=['FUND_NAME'])

#Set FRONT and REAR Loads NaN entries to zero
values = {'REAR_LOAD': 0, 'FRONT_LOAD': 0}
df = df.fillna(value=values)

#Only funds with FRONT_LOAD & REAR_LOAD of zero
df = df[df.FRONT_LOAD == 0]
df = df[df.REAR_LOAD == 0]

#Drop FRONT_LOAD and REAR_LOAD Columns
df = df.drop(['FRONT_LOAD', 'REAR_LOAD'], axis=1)

#Only funds over 35 months in age 
df = df[df.Age > 35]

#Only funds over 70% in Equities
df = df[df.ALL_EQ > 70]

#US funds filter step 01 according to CRSP documentation
df = df[df['CRSP_OBJ_CD'].str.startswith('ED', na = False)]
#Filter out funds with hedge fund strategies according to CRSP documentation
hedge_strategies = ['EDYH', 'EDYS']
df = df[~df['CRSP_OBJ_CD'].isin(hedge_strategies)]

#Filter out all index funds
#-----------------------------------------#

#Create a copy to avoid potential errors with original dataframe
df1 = df.copy()
#Convert the values in the CLADT to datetime
df1.loc[:, 'CALDT'] = pd.to_datetime(df['CALDT'])

#Now create the two dataframes before and after the 'D' index was introduced
df1_1 = df1[df1['CALDT'] < pd.to_datetime('2003-06-30')]
df1_2 = df1[df1['CALDT'] >= pd.to_datetime('2003-06-30')]
#Filter out index funds from dataframe which has index fund flag
df1_3 = df1_2[df1_2.INDEX_FUND_FLAG != 'D']
#Filer out index funds prior to index fund flag
dfx = df1_1.drop(df1_1[df1_1.FUND_NAME.str.contains('Index')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('Idx')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('Ix')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('Indx')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('NASDAQ')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('Nasdaq')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('Dow')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('Mkt')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('DJ')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('S&P 500')].index)
dfx = dfx.drop(dfx[dfx.FUND_NAME.str.contains('BARRA')].index)
#Concatenate the two dataframes back together
df1 = pd.concat([dfx, df1_3], axis=0)

#Create boolean columns which return whether fund is managed by team or single manager
df1 = df1.reindex(df1.columns.tolist() + ['IS_TEAM','NOT_TEAM'], axis=1)
#Create dummy variable for IS_TEAM
df1['IS_TEAM'] = np.where(df1['MANAGER_MSTAR'].str.contains('/', na = False) == True, 1, 0)
#Create dummy variable for NOT_TEAM
df1['NOT_TEAM'] = np.where(df1['IS_TEAM'] == 1, 0, 1)
#Drop MANAGER_MSTAR column
df1 = df1.drop('MANAGER_MSTAR', axis=1)

#Filter out fund rows prior to them reaching 5 million in assets under management
#-----------------------------------------#

#Create a list with dataframes for each fund 
portfolio_NO = df1["CRSP_FUNDNO"].unique()
list_filtering = [df1[df1["CRSP_FUNDNO"] == i].reset_index(drop = True) for i in tqdm(portfolio_NO)]

#Delete observations 
portfolios_sample = []

for df_ in tqdm(list_filtering):
    tmp = df_
    if tmp["MTNA"].max() >= 5:
        mask = tmp[tmp["MTNA"] >= 5]
        portfolios_sample.append(tmp.iloc[mask.index[0]:])

#Combine sample again
sample = pd.concat(portfolios_sample, axis = 0)
    
#Make a copy of sample as df15
df2 = sample.copy()

#Data preparation for ML models
#Drop unnecessary columns
drop_columns = ['FUND_NAME', 'INDEX_FUND_FLAG', 'ET_FLAG', 'CRSP_OBJ_CD', 'flows', 'FIRST_OFFER_DT', 'TERMINATION_DT']
df2 = df2.drop(columns=drop_columns)

#Sort the dates column from lowest to highest
df2 = df2.sort_values(by = 'CALDT')

#Shift the t_alpha_FF5MOM back by one month for prediciton
df2['alpha_FF5MOM_shifted'] = df2.groupby('CRSP_FUNDNO')['alpha_FF5MOM'].shift(-1)


#Drop NaN values from dataframe for linear regression
df2 = df2.dropna()

#Split data into training and test data
df2_1 = df2[df2['CALDT'] < pd.to_datetime('2015-01-01')]
df2_2 = df2[df2['CALDT'] >= pd.to_datetime('2015-01-01')]

#Drop CLADT columns from the new dataframes
training_data = df2_1.drop(['CALDT'], axis = 1)
testing_data = df2_2.drop(['CALDT'], axis = 1)

#Create the funcion to test for normality in the residuals
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    #Plot histogram of residuals
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.show()
    
    #Plot Q-Q plot of residuals
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q plot of Residuals')
    plt.show()
    

#mean absolute percentage error function    
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

#-----------------------------------------#
#Linear Regression Model

#Define independent and dependent variables
x_train = training_data.drop('alpha_FF5MOM_shifted', axis = 1)
y_train = training_data['alpha_FF5MOM_shifted']
x_test = testing_data.drop('alpha_FF5MOM_shifted', axis = 1)
y_test = testing_data["alpha_FF5MOM"]

#Create regression object
reg_OLS = LinearRegression()

#Fit reg to the data
reg_OLS.fit(x_train, y_train)

#Print model coefficients
print('Coefficients: ', reg_OLS.coef_)
print('Intercept: ', reg_OLS.intercept_)

#Predict the target variable on the testing data
y_pred_OLS = reg_OLS.predict(x_test)

#Compute the mean squared error between the true and predicted values
mse_linear_regression_unadjusted = mean_squared_error(y_test, y_pred_OLS)

#Calculate R-Squared value for testing data
r2_linear_regression_unadjusted = reg_OLS.score(x_test, y_test)

#Calculate adjusted R-squared value for testing data
n = len(y_test)
k = x_test.shape[1]
adj_r2_linear_regression_unadjusted = 1 - (1 - r2_linear_regression_unadjusted ) * ((n - 1) / (n - k - 1))

#Plot residuals for Linear Regression Model
plot_residuals(y_test, y_pred_OLS)

#Return mape score
mape_linear_regression_unadjusted = mape(y_test, y_pred_OLS)

#mean absolute error
mae_linear_regression_unadjusted = mean_absolute_error(y_test, y_pred_OLS)

#-----------------------------------------#
#Linear Regression Model - Adjusted

#independent variables for linear regression
drop_columns2 = ["alpha_FF5MOM_shifted", "ALL_EQ", "flows_wins", "flows_vola", "excess_return", "vola_return","alpha_FFC", "t_alpha_FFC", "value_added", "FUND_TNA"]
x_train_OLS = training_data.drop(columns=drop_columns2)
x_test_OLS = testing_data.drop(columns=drop_columns2)

#Create regression object
reg_OLS_2 = LinearRegression()

#Fit reg to the data
reg_OLS_2.fit(x_train_OLS, y_train)

#Print model coefficients
print('Coefficients: ', reg_OLS_2.coef_)
print('Intercept: ', reg_OLS_2.intercept_)

#Predict the target variable on the testing data
y_pred_OLS_2 = reg_OLS_2.predict(x_test_OLS)

#Compute the mean squared error between the true and predicted values
mse_linear_regression_adjusted_01 = mean_squared_error(y_test, y_pred_OLS_2)

#Calculate R-Squared value for testing data
r2_linear_regression_adjusted_01  = reg_OLS_2.score(x_test_OLS, y_test)

#Calculate adjusted R-squared value for testing data
n = len(y_test)
k_OLS = x_test_OLS.shape[1]
r2_adj_linear_regression_adjusted_01 = 1 - (1 - r2_linear_regression_adjusted_01) * ((n - 1) / (n - k_OLS - 1))

#Plot residuals for Linear Regression Model
plot_residuals(y_test, y_pred_OLS_2)

#Return mape score
mape_linear_regression_adjusted = mape(y_test, y_pred_OLS_2)

#mean absolute error
mae_linear_regression_adjusted = mean_absolute_error(y_test, y_pred_OLS_2)

#-----------------------------------------#
#Linear Regression Model - Adjusted 02

#independent variables for linear regression
keep_columns = ["alpha_FF5MOM_shifted", "ALL_EQ", "flows_wins", "alpha_FF5MOM", "flows_vola", "excess_return", "vola_return","alpha_FFC", "t_alpha_FFC", "value_added", "FUND_TNA"]
#for new regression
df_new_reg_train = training_data.loc[:, keep_columns]
df_new_reg_test = testing_data.loc[:, keep_columns]

x_train_OLS_adj_02 = df_new_reg_train.drop("alpha_FF5MOM_shifted", axis = 1)
y_train_OLS_adj_02 = df_new_reg_train["alpha_FF5MOM_shifted"]
x_test_OLS_adj_02 = df_new_reg_test.drop('alpha_FF5MOM_shifted', axis = 1)
y_test_new_reg = df_new_reg_test["alpha_FF5MOM"]

#Create regression object
reg_OLS3 = LinearRegression()

#Fit reg to the data
reg_OLS3.fit(x_train_OLS_adj_02, y_train_OLS_adj_02)
#Print model coefficients
print('Coefficients: ', reg_OLS3.coef_)
print('Intercept: ', reg_OLS3.intercept_)

#Predict the target variable on the testing data
y_pred_OLS3 = reg_OLS3.predict(x_test_OLS_adj_02)

#Compute the mean squared error between the true and predicted values
mse_lin_regression_adjusted_02 = mean_squared_error(y_test_new_reg, y_pred_OLS3)

#Calculate R-Squared value for testing data
r2_lin_regression_adjusted_02 = reg_OLS3.score(x_test_OLS_adj_02, y_test_new_reg)

#Calculate adjusted R-squared value for testing data
n = len(y_test_new_reg)
k = x_test_OLS_adj_02.shape[1]
adj_r2_tlin_regression_adjusted_02 = 1 - (1 - r2_lin_regression_adjusted_02) * ((n - 1) / (n - k - 1))

#Plot residuals for Linear Regression Model
plot_residuals(y_test_new_reg, y_pred_OLS3)

#Return mape score
mape_lin_regression_adjusted_02 = mape(y_test_new_reg, y_pred_OLS3)

#mean absolute error
mae_lin_regression_adjusted_02 = mean_absolute_error(y_test_new_reg, y_pred_OLS3)

#-----------------------------------------#
# Gradient Boosting with hyperparameters tunning:
    
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search over
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'max_iter': [100, 500, 1000],
    'max_depth': [5, 10, 50, 100]
}

# Create HistGradientBoostingRegressor object
reg_GradientBoosting = HistGradientBoostingRegressor(random_state=42)

# Use GridSearchCV to search over the parameter grid
grid_search_GradientBoosting = GridSearchCV(reg_GradientBoosting, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_GradientBoosting.fit(x_train, y_train)

# Get the best model
best_model_GradientBoosting = grid_search_GradientBoosting.best_estimator_

# Fit the best model to the data
best_model_GradientBoosting.fit(x_train, y_train)

# Predict the target variable on testing data
y_pred_GradientBoosting = best_model_GradientBoosting.predict(x_test)

# Compute the mean squared error between the true and predicted values
mse_gradient_boosting = mean_squared_error(y_test, y_pred_GradientBoosting)

# Calculate R-Squared value for testing data
r2_gradient_boosting = best_model_GradientBoosting.score(x_test, y_test)

# Calculate adjusted R-squared value for testing data
n = len(y_test)
k = x_test.shape[1]
adj_r2_gradient_boosting = 1 - ((1 - r2_gradient_boosting) * (n - 1) / (n - k - 1))

#Plot residuals for Gradient Boosting Model
plot_residuals(y_test, y_pred_GradientBoosting)

#Return mape score
mape_gradient_boosting = mape(y_test, y_pred_GradientBoosting)

#mean absolute error
mae_gradient_boosting = mean_absolute_error(y_test, y_pred_GradientBoosting)


#-----------------------------------------#
#Random Forest
from sklearn.ensemble import RandomForestRegressor

RandFor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=3)

RandFor.fit(x_train, y_train)

y_pred_RandFor = RandFor.predict(x_test)

mse_random_forest_base_model = mean_squared_error(y_test, y_pred_RandFor)
r2_random_forest_base_model = r2_score(y_test, y_pred_RandFor)
adj_r2_random_forest_base_model = 1- ((1-r2_random_forest_base_model) * (n-1) / (n-k-1))

print("Mean Squarred Error: " ,mse_random_forest_base_model)
print("R squarred: " , r2_random_forest_base_model)

#Plot residuals for adjusted Random Forest model 01
plot_residuals(y_test, y_pred_RandFor)

#Return mape score
mape_random_forest_base_model = mape(y_test, y_pred_RandFor)

#mean absolute error
mae_random_forest_base_model = mean_absolute_error(y_test, y_pred_RandFor)

#-----------------------------------------#
#Hyperparameters tunning Random Forest with GridSearchCV:

# Define the random forest model with default hyperparameters
RandFor2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=3)

# Define the hyperparameter grid to search
param_grid = {
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'max_features': [1.0],
    'bootstrap': [False]
}

# Define the grid search object
grid_search_RandFor = GridSearchCV(estimator=RandFor2, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)

# Fit the grid search object to the training data
grid_search_RandFor.fit(x_train, y_train)

# Retrieve the best model and its hyperparameters
best_model_RandFor = grid_search_RandFor.best_estimator_
best_params_RandFor = grid_search_RandFor.best_params_

# Use the best model to make predictions on the test data
y_pred_RandFor2 = best_model_RandFor.predict(x_test)

# Calculate the evaluation metrics
mse_random_forest_grid_search = mean_squared_error(y_test, y_pred_RandFor2)
r2_random_forest_grid_search = r2_score(y_test, y_pred_RandFor2)
adj_r2_random_forest_grid_search = 1 - ((1 - r2_random_forest_grid_search) * (n - 1) / (n - k - 1))

#Plot residuals for adjusted Random Forest model 02
plot_residuals(y_test, y_pred_RandFor2)

#Return mape score
mape_random_forest_grid_search = mape(y_test, y_pred_RandFor2)

#mean absolute error
mae_random_forest_grid_search = mean_absolute_error(y_test, y_pred_RandFor2)

#-----------------------------------------#
#Hyperparameters tunning Random Forest with RandomizedSearchCV:

from sklearn.model_selection import RandomizedSearchCV

# Define the random forest model with default hyperparameters
RandFor3 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=3)

# Define the random search space
random_grid = {
    'max_depth': [10, 30, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'max_features': [1.0],
    'bootstrap': [False]
}

# Define the random search object
random_search = RandomizedSearchCV(estimator=RandFor3, param_distributions=random_grid, n_iter=36, cv=5, n_jobs=-1, verbose=3, random_state=42)

# Fit the random search object to the training data
random_search.fit(x_train, y_train)

# Retrieve the best model and its hyperparameters
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# Use the best model to make predictions on the test data
y_pred_RandFor3 = best_model.predict(x_test)

# Calculate the evaluation metrics
mse_random_forest_ran_grid_search = mean_squared_error(y_test, y_pred_RandFor3)
r2_random_forest_rand_grid_search = r2_score(y_test, y_pred_RandFor3)
adj_r2_random_forest_rand_grid_search = 1 - ((1 - r2_random_forest_rand_grid_search) * (n - 1) / (n - k - 1))

#Plot residuals for adjusted Random Forest model with RandomizedSearchCV
plot_residuals(y_test, y_pred_RandFor3)

#Return mape score
mape_random_forest_ran_grid_search = mape(y_test, y_pred_RandFor3)

#mean absolute error
mae_random_forest_ran_grid_search = mean_absolute_error(y_test, y_pred_RandFor3)


#Linear Regression Model - Adjsuted 02
#-----------------------------------------#

#independent variables for linear regression
keep_columns = ["alpha_FF5MOM_shifted", "ALL_EQ", "flows_wins", "alpha_FF5MOM", "flows_vola", "excess_return", "vola_return","alpha_FFC", "t_alpha_FFC", "value_added", "FUND_TNA"]
#for new regression
df_new_reg_train = training_data.loc[:, keep_columns]
df_new_reg_test = testing_data.loc[:, keep_columns]

x_train_OLS_adj_02 = df_new_reg_train.drop("alpha_FF5MOM_shifted", axis = 1)
y_train_OLS_adj_02 = df_new_reg_train["alpha_FF5MOM_shifted"]
x_test_OLS_adj_02 = df_new_reg_test.drop('alpha_FF5MOM_shifted', axis = 1)
y_test_new_reg = df_new_reg_test["alpha_FF5MOM"]

#Create regression object
reg_OLS3 = LinearRegression()

#Fit reg to the data
reg_OLS3.fit(x_train_OLS_adj_02, y_train_OLS_adj_02)
#Print model coefficients
print('Coefficients: ', reg_OLS3.coef_)
print('Intercept: ', reg_OLS3.intercept_)

#Predict the target variable on the testing data
y_pred_OLS3 = reg_OLS3.predict(x_test_OLS_adj_02)

#Compute the mean squared error between the true and predicted values
mse_lin_regression_adjusted_02 = mean_squared_error(y_test_new_reg, y_pred_OLS3)

#Calculate R-Squared value for testing data
r2_lin_regression_adjusted_02 = reg_OLS3.score(x_test_OLS_adj_02, y_test_new_reg)

#Calculate adjusted R-squared value for testing data
n = len(y_test_new_reg)
k = x_test_OLS_adj_02.shape[1]
adj_r2_tlin_regression_adjusted_02 = 1 - (1 - r2_lin_regression_adjusted_02) * ((n - 1) / (n - k - 1))

#Plot residuals for Linear Regression Model
plot_residuals(y_test_new_reg, y_pred_OLS3)

#Return mape score
mape_lin_regression_adjusted_02 = mape(y_test_new_reg, y_pred_OLS3)

#mean absolute error
mae_lin_regression_adjusted_02 = mean_absolute_error(y_test_new_reg, y_pred_OLS3)









