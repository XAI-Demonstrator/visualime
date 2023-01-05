"""*BayLIME* (Bayesian Local Interpretable Model-Agnostic Explanations) [1] is an extension of LIME [2]
that exploits prior knowledge and Bayesian reasoning.

[1] Zhao et al. (2021): *BayLIME: Bayesian Local Interpretable Model-Agnostic Explanations.*
`arXiv:2012.03058 <https://arxiv.org/abs/2012.03058>`_

[2] Ribeiro et al. (2016): *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.*
`arXiv:1602.04938 <https://arxiv.org/abs/1602.04938>`_
"""
from warnings import warn

from sklearn.linear_model import BayesianRidge


class BayesianRidgeFixedLambda(BayesianRidge):
    """BayesianRidge model with fixed parameter lambda.
    This is equivalent to the "partial informative priors" option in the BayLIME framework.

    The value for `lambda_init` will be treated as a constant.
    The parameter `epsilon` specifies the maximum amount of change allowed for `lambda`.
    A warning will be issued if this amount is exceeded.
    To reduce the amount of change in lambda, increase `large_number`.

    See the documentation for :meth:`sklearn.linear_model.BayesianRidge` for a list
    of available parameters.
    """

    def __init__(self, *, epsilon: float = 1e-6, large_number: float = 1e9, **kwargs):
        if "lambda_init" not in kwargs:
            raise ValueError("'lambda_init' must be set.")

        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.large_number = large_number

    def fit(self, X, y, sample_weight=None):
        self.lambda_1 = self.lambda_init * self.large_number
        self.lambda_2 = self.large_number

        super().fit(X, y, sample_weight)

        if abs(self.lambda_ - self.lambda_init) > self.epsilon:
            warn(
                f"lambda changed by more than {self.epsilon:.2E}. See documentation for details.",
                UserWarning,
            )

    fit.__doc__ = BayesianRidge.fit.__doc__


class BayesianRidgeFixedAlphaLambda(BayesianRidge):
    """BayesianRidge model with fixed parameters alpha and lambda.
    This is equivalent to the "full informative priors" option in the BayLIME framework.

    The values for `alpha_init` and `lambda_init` will be treated as constants.

    The parameter `epsilon` specifies the maximum amount of change allowed
    for `alpha` and `lambda`.
    A warning will be issued if this amount is exceeded.
    To reduce the amount of change, increase `large_number`.

    See the documentation for :meth:`sklearn.linear_model.BayesianRidge` for a list
    of available parameters.
    """

    def __init__(self, *, epsilon: float = 1e-6, large_number: float = 1e9, **kwargs):
        if "alpha_init" not in kwargs:
            raise ValueError("'alpha_init' must be set.")
        if "lambda_init" not in kwargs:
            raise ValueError("'lambda_init' must be set.")

        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.large_number = large_number

    def fit(self, X, y, sample_weight=None):
        # In principle, it would be sufficient to set self.n_iter = 0 to skip the convergence loop,
        # but then we get an UnboundLocalError due to the implementation of fit()
        self.n_iter = 1

        self.lambda_1 = self.lambda_init * self.large_number
        self.lambda_2 = self.large_number

        self.alpha_1 = self.alpha_init * self.large_number
        self.alpha_2 = self.large_number

        super().fit(X, y, sample_weight)

        if abs(self.lambda_ - self.lambda_init) > self.epsilon:
            warn(
                f"lambda changed by more than {self.epsilon:.2E}. See documentation for details.",
                UserWarning,
            )

        if abs(self.alpha_ - self.alpha_init) > self.epsilon:
            warn(
                f"alpha changed by more than {self.epsilon:.2E}. See documentation for details.",
                UserWarning,
            )

    fit.__doc__ = BayesianRidge.fit.__doc__
