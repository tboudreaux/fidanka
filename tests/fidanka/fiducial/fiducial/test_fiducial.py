import pandas as pd
import pickle as pkl
import numpy as np
import os

FIDANKA_ROOT = os.environ.get("FIDANKA_TEST_ROOT_DIR")

assert FIDANKA_ROOT is not None, "Please set FIDANKA_TEST_ROOT_DIR environment variable"

PHOTCSV = os.path.join(FIDANKA_ROOT, "input/photometry.csv")

MODULEDIR = os.path.join(FIDANKA_ROOT, "fiducial/fiducial/")

TARGETDIR = os.path.join(MODULEDIR, "target")
INPUTDIR = os.path.join(MODULEDIR, "input")


def test_hull_density():
    inputPhotometry = pd.read_csv(PHOTCSV)

    color = inputPhotometry["F275W"] - inputPhotometry["F814W"]
    mag = inputPhotometry["F814W"]

    from fidanka.fiducial.fiducial import hull_density

    density = hull_density(color, mag, n=50)

    with open(os.path.join(TARGETDIR, "target-hull_density.pkl"), "rb") as f:
        target = pkl.load(f)

    assert (density == target).all()


def test_instantanious_hull_density():
    inputPhotometry = pd.read_csv(PHOTCSV)

    color = inputPhotometry["F275W"] - inputPhotometry["F814W"]
    mag = inputPhotometry["F814W"]

    from fidanka.fiducial.fiducial import instantaious_hull_density

    r0 = np.array([color[500], mag[500]])
    ri = np.vstack([color, mag]).T
    density = instantaious_hull_density(r0, ri, n=50)

    with open(
        os.path.join(TARGETDIR, "target-instantanious_hull_density.pkl"), "rb"
    ) as f:
        target = pkl.load(f)

    assert target[0] == density[0]
    assert (target[2] == density[2]).all()


def test_color_mag_from_filter():
    inputPhotometry = pd.read_csv(PHOTCSV)

    from fidanka.fiducial.fiducial import color_mag_from_filters

    color, mag = color_mag_from_filters(
        inputPhotometry["F275W"], inputPhotometry["F814W"], False
    )
    colorR, magR = color_mag_from_filters(
        inputPhotometry["F275W"], inputPhotometry["F814W"], True
    )

    with open(os.path.join(TARGETDIR, "target-color_mag_from_filters.pkl"), "rb") as f:
        target = pkl.load(f)

    colorOkay = np.equal(color, target["color"]).all()
    magOkay = np.equal(mag, target["mag"]).all()
    colorROkay = np.equal(colorR, target["colorR"]).all()
    magROkay = np.equal(magR, target["magR"]).all()

    assert colorOkay and magOkay and colorROkay and magROkay


if __name__ == "__main__":
    test_instantanious_hull_density()
