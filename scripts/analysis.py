# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats
from sklearn.base import BaseEstimator

# Global vars
DATA_PATH = "[PATH TO MAIN DATASET]"
MIN_AUM = 5  # in million
MIN_AGE_MONTHS = 35
MIN_EQ_PERCENT = 70
CRSP_OBJ_CD_PREFIX = "ED"
EXCLUDE_STRATEGIES = {"EDYH", "EDYS"}
INDEX_FUND_DATE_CUTOFF = pd.to_datetime("2003-06-30")
INDEX_PATTERNS = ["Index", "Idx", "Ix", "Indx", "NASDAQ", "Nasdaq", "Dow", "Mkt", "DJ", "S&P 500", "BARRA"]
TRAIN_TEST_SPLIT_DATE = pd.to_datetime("2015-01-01")
DROP_COLUMNS_PRE_ML = ['FUND_NAME', 'INDEX_FUND_FLAG', 'ET_FLAG', 'CRSP_OBJ_CD', 'flows', 'FIRST_OFFER_DT', 'TERMINATION_DT']
DROP_COLUMNS_ADJUSTED_MODEL = ["alpha_FF5MOM_shifted", "ALL_EQ", "flows_wins", "flows_vola", "excess_return", "vola_return","alpha_FFC", "t_alpha_FFC", "value_added", "FUND_TNA"]
KEEP_COLUMNS_ADJUSTED_MODEL_2 = ["alpha_FF5MOM_shifted", "ALL_EQ", "flows_wins", "alpha_FF5MOM", "flows_vola", "excess_return", "vola_return","alpha_FFC", "t_alpha_FFC", "value_added", "FUND_TNA"]

# Helpers

def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Takes the path to the main dataset and performs the necessary cleansing.
    We filter for various criteria on the CRSP data to eventually only include
    actively managed mutual funds.

    Args:
        path (str): Path to locally hosted data

    Returns:
        pd.DataFrame: Mutual fund data with initial filtering applied
    """
    df = pd.read_csv(path, delimiter=" ", low_memory=False)
    df = df.dropna(subset=['FUND_NAME'])
    df.fillna({'REAR_LOAD': 0, 'FRONT_LOAD': 0}, inplace=True)
    df = df[(df.FRONT_LOAD == 0) & (df.REAR_LOAD == 0)]
    df.drop(columns=['FRONT_LOAD', 'REAR_LOAD'], inplace=True)
    df = df[df.Age > MIN_AGE_MONTHS]
    df = df[df.ALL_EQ > MIN_EQ_PERCENT]
    df = df[df['CRSP_OBJ_CD'].str.startswith(CRSP_OBJ_CD_PREFIX, na=False)]
    df = df[~df['CRSP_OBJ_CD'].isin(EXCLUDE_STRATEGIES)]
    return df

def filter_index_funds(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is a further step in the data cleansing process to remove
    all the remaining index funds from the full fund dataset. We do this separately from
    the main cleansing logic when the dataset initially loads.

    Args:
        df (pd.DataFrame): The main dataframe containing all the CRSP data

    Returns:
        pd.DataFrame: Dataset further cleansed of index fund data
    """
    df['CALDT'] = pd.to_datetime(df['CALDT'])
    before_cutoff = df[df['CALDT'] < INDEX_FUND_DATE_CUTOFF]
    after_cutoff = df[df['CALDT'] >= INDEX_FUND_DATE_CUTOFF]

    # Filter after cutoff by INDEX_FUND_FLAG != 'D'
    after_cutoff = after_cutoff[after_cutoff.INDEX_FUND_FLAG != 'D']

    # Filter before cutoff by fund name patterns
    mask = pd.Series(True, index=before_cutoff.index)
    for pattern in INDEX_PATTERNS:
        # dynamically update the boolean mask
        # https://stackoverflow.com/questions/21237767/python-a-b-meaning
        mask &= ~before_cutoff['FUND_NAME'].str.contains(pattern, na=False, case=False)
    before_cutoff = before_cutoff[mask]

    filtered_df = pd.concat([before_cutoff, after_cutoff], axis=0)
    return filtered_df

def add_team_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function enriches the dataset by adding an additional flag as
    to whether the actively managed mutual fund is managed by a single
    manager or a team of managers and what effect this has on performance.

    Args:
        df (pd.DataFrame): The full CRSP dataset

    Returns:
        pd.DataFrame: Dataset with team flag
    """
    df['IS_TEAM'] = np.where(df['MANAGER_MSTAR'].str.contains('/', na=False), 1, 0)
    df['NOT_TEAM'] = 1 - df['IS_TEAM']
    df.drop(columns=['MANAGER_MSTAR'], inplace=True)
    return df

def filter_min_aum(df: pd.DataFrame, min_aum: float = MIN_AUM) -> pd.DataFrame:
    """
    This function further filters the CRSP dataset to only include funds that have
    at least MIN_AUM assets under management.

    Args:
        df (pd.DataFrame): The full CRSP dataset
        min_aum (float, optional): The threshold for inclusion, in millions of USD. Defaults to MIN_AUM.

    Returns:
        pd.DataFrame: Filtered dataset with only relevant 
    """
    unique_funds = df['CRSP_FUNDNO'].unique()
    filtered_funds = []

    for fund_no in tqdm(unique_funds, desc="Filtering funds by AUM"):
        fund_df = df[df['CRSP_FUNDNO'] == fund_no].reset_index(drop=True)
        if fund_df['MTNA'].max() >= min_aum:
            idx_first = fund_df[fund_df['MTNA'] >= min_aum].index[0]
            filtered_funds.append(fund_df.iloc[idx_first:])

    filtered_df = pd.concat(filtered_funds, axis=0)
    return filtered_df

def prepare_ml_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes the filtered dataset and returns the training
    and testing data based on the date, since we need to keep the temporal
    nature of the data intact for the purposes of this analysis.

    Args:
        df (pd.DataFrame): The filtered CRSP dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing data
    """
    df = df.drop(columns=DROP_COLUMNS_PRE_ML)
    df = df.sort_values('CALDT')
    df['alpha_FF5MOM_shifted'] = df.groupby('CRSP_FUNDNO')['alpha_FF5MOM'].shift(-1)
    df = df.dropna()

    train_df = df[df['CALDT'] < TRAIN_TEST_SPLIT_DATE].drop(columns=['CALDT'])
    test_df = df[df['CALDT'] >= TRAIN_TEST_SPLIT_DATE].drop(columns=['CALDT'])
    return train_df, test_df

def plot_residuals(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Function to generate a probability plot of the residuals of a given model.

    Args:
        y_true (pd.Series): Series of the true alpha values for the testing set
        y_pred (pd.Series): The model's predicted values
    
    Returns:
        Doesn't return a value
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')

    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q plot of Residuals')

    plt.tight_layout()
    plt.show()

def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Function that calculates the Mean Absolute Percentage Error (MAPE) between actual and predicted values.

    Args:
        y_true (pd.Series): Array of true target values.
        y_pred (pd.Series): Array of predicted target values.

    Returns:
        float: The MAPE value, expressed as a decimal (e.g., 0.1 for 10%).
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def calculate_adj_r2(r2: float, n: int, k: int) -> float:
    """
    Calculates the adjusted R-squared value.

    Args:
        r2 (float): The R-squared value.
        n (int): The number of observations.
        k (int): The number of independent variables.

    Returns:
        float: The adjusted R-squared value.
    """
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Model logic

def train_linear_regression(x_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    This function creates and trains a basic linear regression model. The model 
    is purposely untuned and left "out of the box."

    Args:
        x_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.

    Returns:
        LinearRegression: Trained Linear Regression model.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(
    model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[pd.Series, float, float, float, float, float]:
    """
    Evaluate a trained model and compute performance metrics.

    Args:
        model (BaseEstimator): A trained scikit-learn model.
        x_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): True target values for the test set.

    Returns:
        Tuple containing:
            - y_pred (pd.Series): Predicted values.
            - mse (float): Mean Squared Error.
            - r2 (float): R-squared score.
            - adj_r2 (float): Adjusted R-squared score.
            - mape_val (float): Mean Absolute Percentage Error.
            - mae (float): Mean Absolute Error.
    """
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = calculate_adj_r2(r2, len(y_test), x_test.shape[1])
    mape_val = mape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mse, r2, adj_r2, mape_val, mae

def run_linear_regression_models(training_data, testing_data):
    print("\n===== Linear Regression Model (Unadjusted) =====")
    x_train = training_data.drop('alpha_FF5MOM_shifted', axis=1)
    y_train = training_data['alpha_FF5MOM_shifted']
    x_test = testing_data.drop('alpha_FF5MOM_shifted', axis=1)
    y_test = testing_data['alpha_FF5MOM']

    model = train_linear_regression(x_train, y_train)
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    y_pred, mse, r2, adj_r2, mape_val, mae = evaluate_model(model, x_test, y_test)
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}, Adj R2: {adj_r2:.4f}, MAPE: {mape_val:.4f}, MAE: {mae:.4f}")
    plot_residuals(y_test, y_pred)

    # Adjusted Model
    print("\n===== Linear Regression Model (Adjusted) =====")
    x_train_adj = training_data.drop(columns=DROP_COLUMNS_ADJUSTED_MODEL)
    x_test_adj = testing_data.drop(columns=DROP_COLUMNS_ADJUSTED_MODEL)

    model_adj = train_linear_regression(x_train_adj, y_train)
    print(f"Coefficients: {model_adj.coef_}")
    print(f"Intercept: {model_adj.intercept_}")

    y_pred_adj, mse_adj, r2_adj, adj_r2_adj, mape_adj, mae_adj = evaluate_model(model_adj, x_test_adj, y_test)
    print(f"MSE: {mse_adj:.4f}, R2: {r2_adj:.4f}, Adj R2: {adj_r2_adj:.4f}, MAPE: {mape_adj:.4f}, MAE: {mae_adj:.4f}")
    plot_residuals(y_test, y_pred_adj)

    # Adjusted Model 2 - keep columns
    print("\n===== Linear Regression Model (Adjusted 2) =====")
    train_sub = training_data.loc[:, KEEP_COLUMNS_ADJUSTED_MODEL_2]
    test_sub = testing_data.loc[:, KEEP_COLUMNS_ADJUSTED_MODEL_2]

    x_train_adj2 = train_sub.drop("alpha_FF5MOM_shifted", axis=1)
    y_train_adj2 = train_sub["alpha_FF5MOM_shifted"]
    x_test_adj2 = test_sub.drop("alpha_FF5MOM_shifted", axis=1)
    y_test_adj2 = test_sub["alpha_FF5MOM"]

    model_adj2 = train_linear_regression(x_train_adj2, y_train_adj2)
    print(f"Coefficients: {model_adj2.coef_}")
    print(f"Intercept: {model_adj2.intercept_}")

    y_pred_adj2, mse_adj2, r2_adj2, adj_r2_adj2, mape_adj2, mae_adj2 = evaluate_model(model_adj2, x_test_adj2, y_test_adj2)
    print(f"MSE: {mse_adj2:.4f}, R2: {r2_adj2:.4f}, Adj R2: {adj_r2_adj2:.4f}, MAPE: {mape_adj2:.4f}, MAE: {mae_adj2:.4f}")
    plot_residuals(y_test_adj2, y_pred_adj2)

def run_gradient_boosting(x_train, y_train, x_test, y_test):
    print("\n===== Gradient Boosting Regressor with Hyperparameter Tuning =====")

    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'max_iter': [100, 500, 1000],
        'max_depth': [5, 10, 50, 100]
    }

    reg = HistGradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(reg, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")

    y_pred = best_model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = calculate_adj_r2(r2, len(y_test), x_test.shape[1])
    mape_val = mape(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MSE: {mse:.4f}, R2: {r2:.4f}, Adj R2: {adj_r2:.4f}, MAPE: {mape_val:.4f}, MAE: {mae:.4f}")
    plot_residuals(y_test, y_pred)

def run_random_forest_models(x_train, y_train, x_test, y_test):
    print("\n===== Random Forest Regressor (Base Model) =====")
    rf_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=3)
    rf_base.fit(x_train, y_train)

    y_pred_base = rf_base.predict(x_test)
    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)
    adj_r2_base = calculate_adj_r2(r2_base, len(y_test), x_test.shape[1])
    mape_base = mape(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_pred_base)

    print(f"MSE: {mse_base:.4f}, R2: {r2_base:.4f}, Adj R2: {adj_r2_base:.4f}, MAPE: {mape_base:.4f}, MAE: {mae_base:.4f}")
    plot_residuals(y_test, y_pred_base)

    print("\n===== Random Forest Regressor with GridSearchCV =====")
    param_grid = {
        'max_depth': [10, 30, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'max_features': [1.0],
        'bootstrap': [False]
    }

    rf_gs = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=3)
    grid_search = GridSearchCV(rf_gs, param_grid, cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(x_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"Best Params (GridSearch): {grid_search.best_params_}")

    y_pred_gs = best_rf.predict(x_test)
    mse_gs = mean_squared_error(y_test, y_pred_gs)
    r2_gs = r2_score(y_test, y_pred_gs)
    adj_r2_gs = calculate_adj_r2(r2_gs, len(y_test), x_test.shape[1])
    mape_gs = mape(y_test, y_pred_gs)
    mae_gs = mean_absolute_error(y_test, y_pred_gs)

    print(f"MSE: {mse_gs:.4f}, R2: {r2_gs:.4f}, Adj R2: {adj_r2_gs:.4f}, MAPE: {mape_gs:.4f}, MAE: {mae_gs:.4f}")
    plot_residuals(y_test, y_pred_gs)

    print("\n===== Random Forest Regressor with RandomizedSearchCV =====")
    random_grid = param_grid  # Same as grid for demonstration

    rf_rs = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=3)
    random_search = RandomizedSearchCV(rf_rs, random_grid, n_iter=36, cv=5, n_jobs=-1, verbose=3, random_state=42)
    random_search.fit(x_train, y_train)

    best_rf_rs = random_search.best_estimator_
    print(f"Best Params (RandomizedSearch): {random_search.best_params_}")

    y_pred_rs = best_rf_rs.predict(x_test)
    mse_rs = mean_squared_error(y_test, y_pred_rs)
    r2_rs = r2_score(y_test, y_pred_rs)
    adj_r2_rs = calculate_adj_r2(r2_rs, len(y_test), x_test.shape[1])
    mape_rs = mape(y_test, y_pred_rs)
    mae_rs = mean_absolute_error(y_test, y_pred_rs)

    print(f"MSE: {mse_rs:.4f}, R2: {r2_rs:.4f}, Adj R2: {adj_r2_rs:.4f}, MAPE: {mape_rs:.4f}, MAE: {mae_rs:.4f}")
    plot_residuals(y_test, y_pred_rs)

# ========================== Main Execution ==========================

def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)

    print("Filtering out index funds...")
    df_filtered = filter_index_funds(df)

    print("Adding team manager flags...")
    df_with_flags = add_team_flags(df_filtered)

    print(f"Filtering funds with min AUM >= {MIN_AUM} million...")
    df_min_aum = filter_min_aum(df_with_flags, MIN_AUM)

    print("Preparing data for machine learning models...")
    training_data, testing_data = prepare_ml_data(df_min_aum)

    run_linear_regression_models(training_data, testing_data)

    x_train = training_data.drop('alpha_FF5MOM_shifted', axis=1)
    y_train = training_data['alpha_FF5MOM_shifted']
    x_test = testing_data.drop('alpha_FF5MOM_shifted', axis=1)
    y_test = testing_data['alpha_FF5MOM']

    run_gradient_boosting(x_train, y_train, x_test, y_test)

    run_random_forest_models(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
