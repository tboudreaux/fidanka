import pandas as pd
import pickle as pkl
import numpy as np
import os

PHOTDIR = os.path.join(os.environ.get("FIDANKATESTDIR"), "input/photometry.csv")


def test_hull_density():
    inputPhotometry = pd.read_csv(PHOTDIR)

    color = inputPhotometry["F275W"] - inputPhotometry["F814W"]
    mag = inputPhotometry["F814W"]

    from fidanka.fiducial.fiducial import hull_density

    density = hull_density(color, mag, n=50)

    with open("./target/target-hull_density.pkl", "rb") as f:
        target = pkl.load(f)

    assert (density == target).all()


def test_instantanious_hull_density():
    inputPhotometry = pd.read_csv(PHOTDIR)

    color = inputPhotometry["F275W"] - inputPhotometry["F814W"]
    mag = inputPhotometry["F814W"]

    from fidanka.fiducial.fiducial import instantaious_hull_density

    r0 = np.array([color[500], mag[500]])
    ri = np.vstack([color, mag]).T
    density = instantaious_hull_density(r0, ri, n=50)

    with open("./target/target-instantanious_hull_density.pkl", "rb") as f:
        target = pkl.load(f)

    assert target[0] == density[0]
    assert (target[2] == density[2]).all()


if __name__ == "__main__":
    test_instantanious_hull_density()
