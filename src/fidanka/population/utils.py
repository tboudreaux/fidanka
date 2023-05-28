import numpy as np
from scipy import interpolate as interp

def IMF(m=None, alpha=None):
    assert m is not None, "Mass must be specified."
    assert alpha is not None, "Power law index must be specified."

    return m**(-alpha)

def joint_distribution(d1, d2, **kwargs):
    """
    Joint distribution of two random variables.
    """
    return d1(**kwargs) * d2(**kwargs)

def inverse_cdf_sample(f=None, y=None, x=None):
    assert f is not None or y is not None, "Either the function or the data must be specified."

    if x is None:
        x = np.linspace(0,1,1000)
    if y is None:
        y = f(x)
    else:
        assert y.shape == x.shape
    cdf_y = np.cumsum(y)
    cdf_y_norm = cdf_y/cdf_y.max()
    inverse_cdf = interp.interp1d(cdf_y_norm, x, bounds_error=False, fill_value='extrapolate')
    return inverse_cdf

def sample_n_masses(n, completness, alpha, mMin=0.1, mMax=2, nNum=100):
    """
    Sample n masses from a joint completness and IMF function.
    """
    mRange = np.linspace(mMin, mMax, nNum)
    y = joint_distribution(d1=IMF, d2=completness, m=mRange, alpha=alpha)
    s = inverse_cdf_sample(y=y, x=mRange)
    return s(np.random.rand(n))


if __name__ == "__main__":
    completness = lambda m, alpha: np.exp(m)
    alpha = 2.35
    print(sample_n_masses(10, completness, alpha))
