import pandas as pd
import pickle as pkl
import numpy as np
import os

import pytest

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


def test_shift_photometry_by_error():
    inputPhotometry = pd.read_csv(PHOTCSV)

    from fidanka.fiducial.fiducial import shift_photometry_by_error

    baseSamplings = np.zeros(shape=len(inputPhotometry)) + 0.5

    mag = inputPhotometry["F814W"]
    magErr = inputPhotometry["F814W_RMS"]
    shiftedPhotometry = shift_photometry_by_error(mag, magErr, baseSamplings)

    with open(
        os.path.join(TARGETDIR, "target-shift_photometry_by_error.pkl"), "rb"
    ) as f:
        # pkl.dump(shiftedPhotometry, f)
        target = pkl.load(f)

    assert (target == shiftedPhotometry).all()


def test_convex_hull_density_approximation_noMC():
    np.random.RandomState(seed=42)

    inputPhotometry = pd.read_csv(PHOTCSV)

    f1 = inputPhotometry["F275W"]
    f2 = inputPhotometry["F814W"]
    err1 = inputPhotometry["F814W_RMS"]
    err2 = inputPhotometry["F814W_RMS"]

    from fidanka.fiducial.fiducial import MC_convex_hull_density_approximation

    density = MC_convex_hull_density_approximation(
        f1,
        f2,
        err1,
        err2,
        reverseFilterOrder=True,
        mcruns=1,
        convexHullPoints=25,
        pbar=False,
    )

    with open(
        os.path.join(TARGETDIR, "target-convex_hull_density_approximation_noMC.pkl"),
        "rb",
    ) as f:
        # pkl.dump(density, f)
        target = pkl.load(f)

    assert (target == density).all()


@pytest.mark.xfail
def test_convex_hull_density_approximation_MC():
    np.random.RandomState(seed=42)

    inputPhotometry = pd.read_csv(PHOTCSV)

    f1 = inputPhotometry["F275W"]
    f2 = inputPhotometry["F814W"]
    err1 = inputPhotometry["F814W_RMS"]
    err2 = inputPhotometry["F814W_RMS"]

    from fidanka.fiducial.fiducial import MC_convex_hull_density_approximation

    density = MC_convex_hull_density_approximation(
        f1,
        f2,
        err1,
        err2,
        reverseFilterOrder=True,
        mcruns=2,
        convexHullPoints=25,
        pbar=False,
    )

    with open(
        os.path.join(TARGETDIR, "target-convex_hull_density_approximation_MC.pkl"), "rb"
    ) as f:
        # pkl.dump(density, f)
        target = pkl.load(f)

    assert (target == density).all()


def test_get_mag_and_color_ranges():
    inputPhotometry = pd.read_csv(PHOTCSV)

    color = inputPhotometry["F275W"] - inputPhotometry["F814W"]
    mag = inputPhotometry["F814W"]

    from fidanka.fiducial.fiducial import get_mag_and_color_ranges

    magRange, colorRange = get_mag_and_color_ranges(color, mag, 0.05, 0.95)
    with open(
        os.path.join(TARGETDIR, "target-get_mag_and_color_ranges.pkl"), "rb"
    ) as f:
        # pkl.dump({"magRange": magRange, "colorRange": colorRange}, f)
        target = pkl.load(f)

    assert target["magRange"] == magRange
    assert target["colorRange"] == colorRange


def test_percentage_within_n_standard_deviations():
    from fidanka.fiducial.fiducial import percentage_within_n_standard_deviations

    p = percentage_within_n_standard_deviations(1)
    assert 68.26894921370859 - 0.000001 < p < 68.26894921370859 + 0.000001


def test_verticalize_CMD():
    np.random.RandomState(seed=42)
    inputPhotometry = pd.read_csv(PHOTCSV)

    f1 = inputPhotometry["F275W"]
    f2 = inputPhotometry["F814W"]
    err1 = inputPhotometry["F814W_RMS"]
    err2 = inputPhotometry["F814W_RMS"]

    from fidanka.fiducial.fiducial import MC_convex_hull_density_approximation

    density = MC_convex_hull_density_approximation(
        f1,
        f2,
        err1,
        err2,
        reverseFilterOrder=True,
        mcruns=1,
        convexHullPoints=25,
        pbar=False,
    )

    color = f1 - f2
    mag = f2

    from fidanka.fiducial.fiducial import verticalize_CMD

    vertColor, ff = verticalize_CMD(
        color, mag, density, binSize="uniformCS", targetStat=35
    )

    with open(os.path.join(TARGETDIR, "target-verticalize_CMD.pkl"), "rb") as f:
        # pkl.dump(vertColor, f)
        target = pkl.load(f)

    okayList = np.zeros(shape=len(vertColor), dtype=bool)
    for idx, (mC, tC) in enumerate(zip(vertColor, target)):
        okay = mC == pytest.approx(tC, rel=0.05)
        okayList[idx] = True

    numTrue = np.sum(okayList)
    total = len(okayList)
    print(f"Number of True: {numTrue}")
    print(f"Total: {total}")
    assert numTrue / total > 0.95


def test_measure_fiducial_line():
    inputPhotometry = pd.read_csv(PHOTCSV)

    f1 = inputPhotometry["F275W"]
    f2 = inputPhotometry["F814W"]
    err1 = inputPhotometry["F814W_RMS"]
    err2 = inputPhotometry["F814W_RMS"]

    from fidanka.fiducial.fiducial import measure_fiducial_lines

    lines = measure_fiducial_lines(
        f1,
        f2,
        err1,
        err2,
        binSize="uniformCS",
        targetStat=25,
        mcruns=1,
        convexHullPoints=25,
        nPops=1,
    )

    with open(os.path.join(TARGETDIR, "target-measure_fiducial_line.pkl"), "rb") as f:
        # pkl.dump(lines, f)
        target = pkl.load(f)

    meanM = lines[0].mean
    meanT = target[0].mean
    print(meanM, meanT)
    okayList = np.zeros(len(meanM[0]), dtype=bool)
    for idx, (mC, tC, mM, tM) in enumerate(zip(meanM[0], meanT[0], meanM[1], meanT[1])):
        okay = mC == pytest.approx(tC, abs=1e-1)
        okay &= mM == pytest.approx(tM, abs=1e-1)
        okayList[idx] = okay

    numTrue = len(okayList[okayList == True])
    total = len(okayList)
    assert numTrue / total >= 0.70, numTrue


if __name__ == "__main__":
    test_verticalize_CMD()
