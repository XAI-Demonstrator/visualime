from typing import Any, Dict, Optional

from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge

LINEAR_MODELS = {
    "ridge": Ridge,
    "bayesian_ridge": BayesianRidge,
    "lasso": Lasso,
    "linear_regression": LinearRegression,
}

MODEL_TYPE_PARAMS_DOC = f"""model_type : str
        The type of linear model to fit.
        Available options are: `"{'"`, `"'.join(list(LINEAR_MODELS.keys())[:-1])}"`,
        and `"{list(LINEAR_MODELS.keys())[-1]}"`.

        See the `scikit-learn documentation
        <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_
        for details on each of the models.

    model_params : dict, optional
        Parameters to pass to the model during instantiation.

        See the `scikit-learn documentation
        <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model>`_
        for details on each of the models."""


def instantiate_model(model_type: str, model_params: Optional[Dict[str, Any]] = None):
    model_params = model_params or {}

    try:
        return LINEAR_MODELS[model_type](**model_params)
    except KeyError:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available options: {', '.join(list(LINEAR_MODELS.keys()))}."
        )
