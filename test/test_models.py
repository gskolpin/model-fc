import math

import numpy as np
from model_fc.models import PearsonRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks([PearsonRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_PearsonRegressor():
    X = np.random.randn(100, 20)
    y = np.random.randn(100)

    pr = PearsonRegressor()
    pr.fit(X, y)
    pred = pr.predict(X)
    assert pred.shape == y.shape


def test_PearsonRegressorScale():
    y = np.random.randn(100)
    X = np.zeros((100, 10))
    for col in range(X.shape[1]):
        # no noise so that our scaling == 1 exactly
        X[:, col] = y

    pr = PearsonRegressor()
    pr.fit(X, y)
    # assert scale is 1/ncols
    assert math.isclose(pr.scale_, 0.1)
