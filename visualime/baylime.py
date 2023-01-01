"""BayLIME extension.

*BayLIME* (Bayesian Local Interpretable Model-Agnostic Explanations) [1] is an extension of LIME [2]
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

    def __init__(self, *, epsilon=1e-6, large_number=1e6, **kwargs):
        if "lambda_init" not in kwargs:
            raise ValueError("'lambda_init' must be set.")
        super(BayesianRidgeFixedLambda, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.large_number = large_number

    def fit(self, X, y, sample_weight=None):
        self.lambda_1 = self.lambda_init * self.large_number
        self.lambda_2 = self.large_number
        super(BayesianRidgeFixedLambda, self).fit(X, y, sample_weight)
        if abs(self.lambda_ - self.lambda_init) > self.epsilon:
            warn(
                f"lambda changed by more than {self.epsilon:.2E}. See documentation for details.",
                RuntimeWarning,
            )

    fit.__doc__ = BayesianRidge.fit.__doc__


class BayesianRidgeFixedAlphaLambda(BayesianRidge):
    """BayesianRidge model with fixed parameters alpha and lambda.
    This is equivalent to the "full informative priors" option in the BayLIME framework.

    The values for `alpha_init` and `lambda_init` will be used as the model's
    parameters, no model selection is performed.

    See the documentation for :meth:`sklearn.linear_model.BayesianRidge` for a list
    of available parameters.
    """

    def __init__(self, **kwargs):
        if "alpha_init" not in kwargs:
            raise ValueError("'alpha_init' must be set.")
        if "lambda_init" not in kwargs:
            raise ValueError("'lambda_init' must be set.")
        super(BayesianRidgeFixedAlphaLambda, self).__init__(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.n_iter = 0
        super(BayesianRidgeFixedAlphaLambda, self).fit(X, y, sample_weight)

    fit.__doc__ = BayesianRidge.fit.__doc__
