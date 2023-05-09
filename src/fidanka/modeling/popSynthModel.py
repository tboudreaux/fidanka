from fidanka.isochrone.isochrone import shift_isochrone
from fidanka.isochrone.isochrone import iso_color_mag
from fidanka.isofit.fit import optimize as isofit

from fidanka.fiducial.fiducial import fiducial_line
from fidanka.fiducial.fiducial import MC_convex_hull_density_approximation
from fidanka.fiducial.fiducial import verticalize_CMD

from scipy.interpolate import interp1d
import numpy as np
import itertools

def calculate_goodness_of_fit(syntheticColor, syntheticDensity, observedColor, observedDensity):
    """
    Calculate the reduced chi-squared goodness-of-fit metric for comparing p(c) between
    the observed and synthetic populations within a given magnitude bin, using linear interpolation.

    Args:
        syntheticColor (array): Array of color values for the synthetic population.
        syntheticDensity (array): Array of density values for the synthetic population.
        observedColor (array): Array of color values for the observed population.
        observedDensity (array): Array of density values for the observed population.

    Returns:
        float: The reduced chi-squared goodness-of-fit metric for the given magnitude bin.
    """

    # Determine which population has fewer data points
    if len(syntheticColor) < len(observedColor):
        x = syntheticColor
        y = syntheticDensity
        x_other = observedColor
        y_other = observedDensity
    else:
        x = observedColor
        y = observedDensity
        x_other = syntheticColor
        y_other = syntheticDensity

    # Sort the color and density data of the population with fewer data points
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Create a linear interpolation function g(c) for the population with fewer data points
    g = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')

    # Evaluate the interpolation function g(c) at each color of the other population
    y_interp = g(x_other)

    # Calculate the differences between the interpolated and other population densities
    differences = y_interp - y_other

    # Calculate the chi-squared values, dividing by the other population density values
    chi_squared = (differences ** 2) / y_other

    # Calculate the reduced chi-squared by dividing the sum of chi-squared values by the degrees of freedom
    reduced_chi_squared = np.sum(chi_squared) / (len(y_other) - 1)

    return reduced_chi_squared


def CMDMagBins(mag, color, iso, bin_size):
    """
    Generate an iterator that yields magnitude bins with their corresponding verticalized color
    and magnitude values using the verticalize_CMD function.

    Args:
        mag (array): Array of magnitude values.
        color (array): Array of color values.
        iso (array): Array representing the isochrone.
        bin_size (float): The size of the magnitude bins.

    Returns:
        iterator: An iterator that yields tuples containing the magnitude bin edges and the
                  verticalized color and magnitude values within the bin.
    """

    # Verticalize the CMD
    verticalizedColor = verticalize_CMD(mag, color, iso)

    # Calculate the minimum and maximum magnitudes to cover the entire range
    min_mag = np.min(mag)
    max_mag = np.max(mag)

    # Generate the bin edges
    bin_edges = np.arange(min_mag, max_mag + bin_size, bin_size)

    # Loop through the magnitude bins
    for i in range(len(bin_edges) - 1):
        # Define the magnitude bin edges
        mag_bin = (bin_edges[i], bin_edges[i + 1])

        # Filter the verticalized color and magnitude data based on the magnitude bin
        color_filtered = verticalizedColor[np.logical_and(mag >= mag_bin[0], mag < mag_bin[1])]
        mag_filtered = mag[np.logical_and(mag >= mag_bin[0], mag < mag_bin[1])]

        yield mag_bin, color_filtered, mag_filtered




def best_fit_dual_iso_pop(
        photometricData,
        isochrones,
        filter1,
        filter2,
        reverseFilterOrder=False,
        binSize=0.1,
        n=100,
        nStars=1000
        ):
    combinations = list(itertools.product(isochrones['A'], isochrones['E']))
    fiducualLine = fiducial_line(
            photometricData[filter1],
            photometricData[filter2],
            photometricData[filter1 + 'RMS'],
            photometricData[filter2 + 'RMS'],
            reverseFilterOrder=reverseFilterOrder,
            mcruns=1
            )

    if reverseFilterOrder:
        filter3 = filter2
    else:
        filter3 = filter1


    # Initialize an empty list to store the results for each valid combination of isochrones
    results = list()
    for isoA, isoE in combinations:

        isoAColor, isoAMag = iso_color_mag(isoA, filter1, filter2, reverseFilterOrder)
        isoEColor, isoEMag = iso_color_mag(isoE, filter1, filter2, reverseFilterOrder)

        # Loop through all pairs of isochrones from different populations
        mu1, E1 = isofit(fiducualLine, isoA, (filter1, filter2, filter3))
        mu2, E2 = isofit(fiducualLine, isoE, (filter1, filter2, filter3))

        # Check if the distance modulus and reddening differ by a significant amount
        if np.abs(mu1 - mu2) > ... and np.abs(E1 - E2) > ...:  # Replace '...' with a configurable threshold
            continue

        # Calculate the average distance modulus and reddening
        mu_mean = (mu1 + mu2) / 2
        E_mean = (E1 + E2) / 2

        # Shift both isochrones by mu_mean and E_mean
        shifted_iso1 = shift_isochrone(isoAColor, isoAMag, mu_mean, E_mean)
        shifted_iso2 = shift_isochrone(isoEColor, isoEMag, mu_mean, E_mean)

        # Generate synthetic populations for both isochrones
        synth_pop1 = generate_synth_pop(shifted_iso1, photometricData["mag_err"], photometricData["color_err"], nStars)
        synth_pop2 = generate_synth_pop(shifted_iso2, photometricData["mag_err"], photometricData["color_err"], nStars)

        # Combine the synthetic populations
        synth_pop = np.vstack((synth_pop1, synth_pop2))

        # Calculate density at each point in the synthetic and observed populations
        synthDensity = MC_convex_hull_density_approximation(synthPop1[filter1], synthPop1[filter2], None, None, reverseFilterOrder=reverseFilterOrder, mcruns=1)
        obsDensity = MC_convex_hull_density_approximation(photometricData[filter1], photometricData[filter2], None, None, reverseFilterOrder=reverseFilterOrder, mcruns=1)

        # Initialize an empty list to store the goodness-of-fit values for each bin
        goodnessOfFitValues = []

        # Loop through the magnitude bins using the CMDMagBins function
        for mag_bin, color_filtered, mag_filtered in CMDMagBins(photometricData["mag"], photometricData["color"], iso1, binSize):
            # Calculate the goodness-of-fit for the current magnitude bin
            goodnessOfFit = calculate_goodness_of_fit(color_filtered, synthDensity, color_filtered, obsDensity)

            # Append the goodness-of-fit value to the list
            goodnessOfFitValues.append(goodnessOfFit)

        # Calculate the total goodness-of-fit
        total_goodness_of_fit = np.sum(goodnessOfFitValues)

        # Save the results for this combination of isochrones
        results.append((total_goodness_of_fit, iso1, iso2))



    # Find the best-fit set of isochrones
    best_fit = min(results, key=lambda x: x[0])

    # Extract the best-fit isochrones and goodness-of-fit value
    best_goodness_of_fit, best_iso1, best_iso2 = best_fit

    return best_goodness_of_fit, best_iso1, best_iso2

#TODO need to add generate_synth_pop function, clean up verticalize feature in the CMDMagBins iterator
