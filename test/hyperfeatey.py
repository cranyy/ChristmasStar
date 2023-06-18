from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

def feature_selection(stock_data, method="SelectKBest", n_features=7):
    X = stock_data.drop("Close", axis=1)
    y = stock_data["Close"]

    if method == "RFE":
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector = selector.fit(X, y)
        selected_columns = X.columns[selector.support_]

    elif method == "SelectKBest":
        selector = SelectKBest(score_func=f_regression, k=n_features)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]

    elif method == "Lasso":
        lasso = Lasso(alpha=0.1)
        lasso.fit(X, y)
        selected_columns = X.columns[lasso.coef_ != 0]

    return selected_columns

def data_has_changed(stock_data_old, stock_data_new):
    return not stock_data_old.equals(stock_data_new)



def score_hyperparams(model, X, y, params, cv_splits):
    model.set_params(**params)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv_splits)
    return params, scores.mean()

class TqdmSplits(KFold):
    def __init__(self, n_splits=5, *args, **kwargs):
        super().__init__(n_splits, *args, **kwargs)

    def split(self, X, y=None, groups=None):
        iterator = super().split(X, y, groups)
        return tqdm(iterator, total=self.n_splits)

def custom_cv_splits(n_splits=3):
    return TqdmSplits(n_splits=n_splits)