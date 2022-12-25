from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge

LINEAR_MODELS = {
    "ridge": Ridge,
    "bayesian_ridge": BayesianRidge,
    "lasso": Lasso,
    "linear_regression": LinearRegression,
}

MODEL_TYPE_DOC = """model_type : str
        The type of linear model to fit.
        Available options are `"bayesian_ridge"`, `"lasso"`, and `"linear_regression"`.

        See the `scikit-learn documentation
        <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_
        for details on each of the methods."""
