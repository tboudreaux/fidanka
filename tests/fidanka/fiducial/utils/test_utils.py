import numpy as np
import pickle as pkl
import pandas as pd
import os

PHOTDIR = os.path.join(os.environ.get("FIDANKATESTDIR"), "input/photometry.csv")


def test_bin_color_mag_density():
    photometry = pd.read_csv(PHOTDIR)

    from fidanka.fiducial.utils import bin_color_mag_density

    color, mag, density = bin_color_mag_density(
        photometry["F275W"] - photometry["F814W"],
        photometry["F814W"],
        photometry["F275W"],
        targetStat=50,
    )

    with open("./target/target-bin_color_mag_density.pkl", "rb") as f:
        target = pkl.load(f)

    colorSame = True
    magSame = True
    densitySame = True
    for c, m, d, tc, tm, td in zip(
        color, mag, density, target["color"], target["mag"], target["density"]
    ):
        colorSame &= np.equal(c, tc).all()
        magSame &= np.equal(m, tm).all()
        densitySame &= np.equal(d, td).all()

    assert colorSame & magSame & densitySame


def test_clean_bins():
    inputData = np.loadtxt("./input/inputData-clean_bins.txt")
    sortedInputIDX = np.argsort(inputData[:, 0])
    sortedInput = inputData[sortedInputIDX]
    bins = np.linspace(sortedInput[:, 0].min(), sortedInput[:, 0].max(), 10)
    binIDs = np.digitize(sortedInput[:, 0], bins)
    binnedDataX = list()
    binnedDataY = list()
    binnedDataZ = list()
    for binID in np.unique(binIDs):
        binnedX = sortedInput[binIDs[binIDs == binID], 0]
        binnedY = sortedInput[binIDs[binIDs == binID], 1]
        binnedZ = sortedInput[binIDs[binIDs == binID], 2]
        binnedDataX.append(binnedX)
        binnedDataY.append(binnedY)
        binnedDataZ.append(binnedZ)

    from fidanka.fiducial.utils import clean_bins

    newX, newY, newZ = clean_bins(
        binnedDataX, binnedDataY, binnedDataZ, sigma=3, iterations=1
    )

    with open("./target/target-clean_bins.pkl", "rb") as f:
        target = pkl.load(f)

    XSame = True
    YSame = True
    ZSame = True
    for nx, ny, nz, tx, ty, tz in zip(
        newX, newY, newZ, target["x"], target["y"], target["z"]
    ):
        XSame &= np.equal(nx, tx).all()
        YSame &= np.equal(ny, ty).all()
        ZSame &= np.equal(nz, tz).all()

    assert XSame & YSame & ZSame


def test_normalize_density_magBin():
    photometry = pd.read_csv(PHOTDIR)
    from fidanka.fiducial.utils import bin_color_mag_density

    color, mag, density = bin_color_mag_density(
        photometry["F275W"] - photometry["F814W"],
        photometry["F814W"],
        photometry["F275W"],
        targetStat=50,
    )


if __name__ == "__main__":
    test_bin_color_mag_density()
