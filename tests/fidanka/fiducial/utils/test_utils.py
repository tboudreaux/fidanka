import numpy as np


def test_clean_bins():
    inputData = np.loadtxt("./inputData.txt")
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
    import pickle as pkl

    with open("target.pkl", "rb") as f:
        target = pkl.load(f)

    XSame = (newX == target["x"]).all()
    YSame = (newY == target["y"]).all()
    ZSame = (newZ == target["z"]).all()

    assert XSame and YSame and ZSame


if __name__ == "__main__":
    test_clean_bins()
