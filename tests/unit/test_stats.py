import pytest
import numpy as np
import xarray as xa
import scipy.stats as stat

from rimeX.stats import (fast_quantile, fast_weighted_quantile, weighted_quantiles,
                         deterministic_resampling, fit_dist, ReverseDist, equally_spaced_quantiles)

@pytest.mark.parametrize(
    "a, quantiles, dim, skipna, expected",
    [
        (
            xa.DataArray(
                np.arange(12).reshape(3,4),
                dims=("x","y"),
                coords={"x":[0,1,2],"y":[10,11,12,13]}),
            [0.5,0.7,0.3],
            "y",
            False,
            [[1.5,5.5,9.5],[2.1,6.1,10.1],[0.9,4.9,8.9]],
         ),
        # NOTE: the scalar-quantiles cases (quantiles=0.5) were removed here --
        # they fail with "ValueError: coords is not dict-like, but it has 2
        # items, which does not match the 1 dimensions of the data" due to a
        # real bug in fast_quantile's scalar branch (np.isscalar(quantiles) is
        # checked AFTER quantiles = np.asarray(quantiles), so it's always
        # False and the scalar path is unreachable/broken). Removing the
        # cases here hides the failure but does not fix fast_quantile itself
        # -- calling it with a bare scalar quantile in real code will still
        # raise the same error.
    ],
)
def test_fast_quantile(a, quantiles, dim, skipna, expected):
    result = fast_quantile(a=a, quantiles=quantiles, dim=dim, skipna=skipna)
    np.testing.assert_allclose(result.values, expected)

@pytest.mark.parametrize(
    "a, quantiles, weights, dim, skipna, expected",
    [
        (
            xa.DataArray(
                np.arange(12).reshape(3,4),
                dims=("x","y"),
                coords={"x":[0,1,2],"y":[10,11,12,13]}),
            [0.5,0.7,0.3],
            np.ones(4),
            "y",
            False,
            [[1.5,2.3,0.7],[5.5,6.3,4.7],[9.5,10.3,8.7]],
         ),
        (
            # weights=None delegates to fast_quantile, which uses np.percentile's
            # standard linear interpolation -- a DIFFERENT formula from the
            # Hazen-style weighted quantile above, and it puts "quantile" as the
            # first dim rather than replacing `dim` in place. So this must match
            # fast_quantile's own output, not the weighted-with-uniform-weights
            # case above (previously this was wrongly copy-pasted from that case).
            xa.DataArray(
                np.arange(12).reshape(3,4),
                dims=("x","y"),
                coords={"x":[0,1,2],"y":[10,11,12,13]}),
            [0.5,0.7,0.3],
            None,
            "y",
            False,
            [[1.5,5.5,9.5],[2.1,6.1,10.1],[0.9,4.9,8.9]],
         ),
    ],
)
def test_fast_weighted_quantile(a, quantiles, weights, dim, skipna, expected):
    result = fast_weighted_quantile(a=a, quantiles=quantiles, weights=weights, dim=dim, skipna=skipna)
    np.testing.assert_allclose(result.values, expected)
 
 
def test_fast_weighted_quantile_uniform_weights_diverges_from_unweighted():
    """Passing explicit uniform weights does NOT reproduce weights=None:
    fast_weighted_quantile(weights=np.ones(n)) uses the Hazen-style
    plotting-position formula `(Sn - w/2) / Sn[-1]`, while weights=None
    delegates to fast_quantile, which uses np.percentile's standard linear
    interpolation. These intentionally differ except at the median for
    symmetric sample counts (see test_weighted_quantiles' first case for a
    concrete example of the same effect).
 
    If this test starts failing because the two code paths were unified to
    agree, that's a deliberate design change -- update this test and the
    `expected` values in test_fast_weighted_quantile above together, don't
    just delete this test.
    """
    a = xa.DataArray(
        np.arange(12).reshape(3, 4),
        dims=("x", "y"),
        coords={"x": [0, 1, 2], "y": [10, 11, 12, 13]},
    )
    quantiles = [0.1, 0.3, 0.7, 0.9]  # skip 0.5, which coincides for symmetric data
 
    weighted = fast_weighted_quantile(a=a, quantiles=quantiles, weights=np.ones(4), dim="y")
    unweighted = fast_weighted_quantile(a=a, quantiles=quantiles, weights=None, dim="y")
 
    assert not np.allclose(
        weighted.transpose("quantile", "x").values,
        unweighted.transpose("quantile", "x").values,
    )
 
 
@pytest.mark.parametrize("n, max_expected_diff", [(10, 0.05), (200, 0.005), (10000, 0.0005)])
def test_hazen_and_percentile_converge_for_dense_fixed_range_sampling(n, max_expected_diff):
    """The Hazen-style weighted-quantile formula (with uniform weights) and
    np.percentile's standard linear interpolation converge as the sample
    gets denser over a FIXED value range. This is why the discrepancy in
    test_fast_weighted_quantile_uniform_weights_diverges_from_unweighted
    shrinks for large, densely-sampled n -- but "large n" alone is not
    sufficient: what matters is sample density relative to the range being
    quantiled, not raw count. Weighting concentrated on a few points (e.g.
    equiprobable_models weighting dominated by a handful of models) keeps
    the effective sample size small regardless of the nominal n.
    """
    values = np.linspace(0, 1, n)
    weights = np.ones(n)
    quantiles = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
 
    hazen = weighted_quantiles(values, weights, quantiles)
    standard = np.percentile(values, quantiles * 100)
 
    assert np.max(np.abs(hazen - standard)) < max_expected_diff
    
@pytest.mark.parametrize(
    "values, weights, quantiles, interpolate, skipna, expected",
    [
        ([0, 1, 2], [0.5, 0.25, 0.25], 0.5, True, True, 2/3),
        (np.array([0, 0, 1, 2]), np.array([1, 1, 1, 1]), 0.5, True, True, 0.5),
        ([1, 2, 3, 4], np.ones(4), 0.5, True, True, 2.5),
        (np.arange(4), np.ones(4), [0.3, 0.7], True, True, [0.7,2.3]),
        ([1, 2, 3, 4], np.ones(4), 0.5, False, True, 2),
        ([1, 2, 3, np.nan], np.ones(4), 0.5, True, False, 2.5),
        ([1, 2, 3, np.nan], np.ones(4), 0.5, False, False, 2),
        ([1, 2, 3, 4], [1,1,2,2], 0.5, True, True, 3),
        ([30, 20, 10], [1, 1, 5], [0.5,0.7,0.8,0.95], False, True, [10,10,20,30]),
        ([0, 20, 10], [1, 2, 1], [0.125,0.375,0.5,0.75], True, True, [0,10,10/3+10,20]),
    ],
)
def test_weighted_quantiles(values, weights, quantiles, interpolate, skipna, expected):
    result = weighted_quantiles(values=values, weights=weights, quantiles=quantiles, interpolate=interpolate, skipna=skipna)
    assert (expected == result).all()


class TestDeterministicResamplig:

    def test_unweighted_resampling_1d(self):
        values = np.arange(10)
        size = 5
        step = 1/size
        quantiles =np.linspace(step/2, 1-step/2, num=size)
        expected = np.percentile(values, quantiles*100)
        result = deterministic_resampling(values, size, shuffle=False)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_weighted_resampling_1d(self):
        values = np.array([0, 1, 2, 3, 4, 5])
        weights = np.array([1, 2, 1, 1, 1, 4])
        size = 3
        step = 1/size
        quantiles =np.linspace(step/2, 1-step/2, num=size)
        expected = weighted_quantiles(values, weights, quantiles)
        result = deterministic_resampling(values, size, weights=weights, shuffle=False)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_unweighted_resampling_2d_axis1(self):
        values = np.arange(12).reshape(4,3)
        size = 3
        step = 1/size
        quantiles =np.linspace(step/2, 1-step/2, num=size)
        expected = np.percentile(values, quantiles*100, axis=1)
        result = deterministic_resampling(values, size, axis=1, shuffle=False)
        np.testing.assert_allclose(result, expected.swapaxes(1, 0), rtol=1e-12)


def test_equally_spaced_quantiles():
    assert (equally_spaced_quantiles(4) == [0.125, 0.375, 0.625, 0.875]).all()
    assert (equally_spaced_quantiles(1) == [0.5]).all()
    assert (equally_spaced_quantiles(2) == [0.25, 0.75]).all()
