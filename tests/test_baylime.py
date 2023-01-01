import numpy as np
import pytest

from visualime.baylime import BayesianRidgeFixedAlphaLambda, BayesianRidgeFixedLambda


def test_that_lambda_fixed():
    X = np.random.rand(100, 20)
    y = np.sum(X * np.random.rand(20), axis=1)

    model = BayesianRidgeFixedLambda(alpha_init=1.5, lambda_init=2.0)

    model.fit(X, y)

    assert abs(model.lambda_ - model.lambda_init) < 1e-6


def test_that_alpha_and_lambda_are_fixed():
    X = np.random.rand(100, 20)
    y = np.sum(X * np.random.rand(20), axis=1)

    model = BayesianRidgeFixedAlphaLambda(alpha_init=1_500, lambda_init=2.0)

    model.fit(X, y)

    assert abs(model.lambda_ - model.lambda_init) < 1e-6
    assert abs(model.alpha_ - model.alpha_init) < 1e-6


@pytest.mark.parametrize(
    "model,kwargs",
    [
        (BayesianRidgeFixedLambda, {}),
        (BayesianRidgeFixedAlphaLambda, {"lambda_init": 5.0}),
        (BayesianRidgeFixedAlphaLambda, {"alpha_init": 0.44}),
    ],
)
def test_that_missing_priors_raise_exception(model, kwargs):
    with pytest.raises(ValueError):
        _ = model(**kwargs)
