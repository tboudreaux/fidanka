import numpy as np

class fiducial_line:
    def __init__(self, name):
        self.name = name
        self._measurements = list()

    def add_measurement(self, color, mag):
        self._measurements.append(np.vstack((color, mag)))

    @property
    def mean(self):
        return np.mean(self._measurements, axis=0)

    @property
    def std(self):
        return np.std(self._measurements, axis=0)

    def confidence(self, p):
        """
        Calculate the upper and lower bound confidence intervals for the mean
        fiducial line.

        Parameters
        ----------
        p : float
            The percentile of the confidence interval. (0 < p < 1)

        Returns
        -------
        lower : ndarray
            The lower bound confidence interval for the mean fiducial line.
        upper : ndarray
            The upper bound confidence interval for the mean fiducial line.

        Examples
        --------
        >>> import numpy as np
        >>> from fiducialLine import fiducial_line
        >>> fl = fiducial_line('test')
        >>> fl.add_measurement(np.array([1, 2, 3]), np.array([4, 5, 6]))
        >>> fl.add_measurement(np.array([2, 3, 4]), np.array([5, 6, 7]))
        >>> fl.add_measurement(np.array([3, 4, 5]), np.array([6, 7, 8]))
        >>> fl.confidence(0.05, 0.95)
        """
        n = len(self._measurements)
        nBin = len(self._measurements[0][0])
        upper = p
        lower = 1 - p

        # Prepare arrays to store lower and upper confidence intervals
        lowerBound = np.zeros(nBin)
        upperBound = np.zeros(nBin)

        # Iterate over each bin
        for i in range(nBin):
            # Collect all color values for this bin across all measurements
            colors = [self._measurements[j][0, i] for j in range(n)]

            # Calculate the lower and upper percentiles
            lowerBound[i] = np.percentile(colors, lower*100)
            upperBound[i] = np.percentile(colors, upper*100)
        CI = np.vstack((lowerBound, upperBound))
        return CI


    def __iter__(self):
        return iter(self._measurements)

    def __repr__(self):
        return f"<Fiducial Line {self.name} : {len(self._measurements)} measurements>"
