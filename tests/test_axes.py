import numpy as np
import pytest

from tact.research.axes import auc, orthogonal_pair, per_problem_difference


def test_paper_difference_weights_problems_equally_not_steps():
    problems = {
        "short": {"labels": np.array(["OVERTHINKING", "CALIBRATED"]),
                  "hidden": np.array([[[2., 0.]], [[0., 0.]]])},
        "long": {"labels": np.array(["OVERTHINKING"] * 3 + ["CALIBRATED"]),
                 "hidden": np.array([[[6., 0.]]] * 3 + [[[0., 0.]]])},
    }
    result = per_problem_difference(problems, "OVERTHINKING", 0)
    np.testing.assert_allclose(result, [4., 0.])


def test_gram_schmidt_axes_preserve_primary_direction():
    ot, oa = orthogonal_pair([2., 0., 0.], [4., 3., 0.])
    np.testing.assert_allclose(ot, [1., 0., 0.])
    np.testing.assert_allclose(oa, [0., 1., 0.])
    assert np.isclose(ot @ oa, 0)
    with pytest.raises(ValueError, match="Degenerate"):
        orthogonal_pair([1., 0.], [2., 0.])


def test_auc_ties_and_separation():
    assert auc([3, 4], [1, 2]) == 1
    assert auc([1, 1], [1, 1]) == 0.5
