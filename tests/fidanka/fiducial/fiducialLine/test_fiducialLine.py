import os
import pytest
import numpy as np
import pandas as pd
import pickle as pkl

FIDANKA_ROOT = os.environ.get("FIDANKA_TEST_ROOT_DIR")

assert FIDANKA_ROOT is not None, "Please set FIDANKA_TEST_ROOT_DIR environment variable"

PHOTCSV = os.path.join(FIDANKA_ROOT, "input/photometry.csv")

MODULEDIR = os.path.join(FIDANKA_ROOT, "fiducial/fiducialLine/")

TARGETDIR = os.path.join(MODULEDIR, "target")
INPUTDIR = os.path.join(MODULEDIR, "input")


# TODO Figure out why this is not being repetable
def test_confidence():
    np.random.seed(42)
    from fidanka.fiducial.fiducialLine import fiducial_line

    fiducial_line("demo")

    with open(os.path.join(TARGETDIR, "target-line_confidence.pkl"), "rb") as f:
        target = pkl.load(f)
    line = target[0]
    tlConfidence = target[1]
    tuConfidence = target[2]
    lConfidence = line.confidence(0.05)
    uConfidence = line.confidence(0.95)

    okayList = list()
    for (l, u), (tl, tu) in zip(
        zip(lConfidence, uConfidence), zip(tlConfidence, tuConfidence)
    ):
        for (ll, uu), (tll, tuu) in zip(zip(l, u), zip(tl, tu)):
            lOkay = ll == pytest.approx(tll, abs=0.3)
            uOkay = uu == pytest.approx(tuu, abs=0.3)
            okayList.append(lOkay and uOkay)

    totalOkay = np.sum(okayList)
    total = len(okayList)
    fraction = totalOkay / total

    print(fraction)
    assert fraction > 0.80


if __name__ == "__main__":
    test_confidence()
