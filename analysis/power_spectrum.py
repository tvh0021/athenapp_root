## Script to calculate the power spectrum given a cubic box
## Written by Trung Ha, with assist from DeepSeek R1:14b

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def parsing():
    datadict = dict()

    parser = argparse.ArgumentParser(description="Power spectrum calculation")
    parser.add_argument(
        "--path",
        type=str,
        default=os.getcwd(),
        help="path to the field data",
        required=False,
    )
    parser.add_argument(
        "--file_name",
        type=str,
        help="name of the file containing velocity/magnetic field data",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="number of bins in the power spectrum calculation",
        required=False,
    )
    args = parser.parse_args()

    file_path = args.path + "/" + args.file_name

    datadict["field"] = np.load(file_path)
    datadict["n_bins"] = args.n_bins

    return datadict


def azimuthal_sum(image: np.ndarray, center=None):
    """
    Calculate the azimuthally summed radial profile.

    cube - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    ny, nx = np.shape(image)

    if not center:
        center = np.array([nx // 2, ny // 2])

    # distances from center
    r = np.hypot(x - center[0], y - center[1])

    # and in indices
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = np.ceil(r_sorted)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius

    # nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=np.float64)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin

    return radial_prof


def compute_spec_3d(cube: np.ndarray, resolution: float = 1):

    nz, ny, nx = np.shape(cube)
    diag_n = np.sqrt(nx**2 + ny**2)
    sp_by_slice = np.empty(
        (nz, int(diag_n // 2))
    )  # the azimuthal_sum function returns a 1d array of length 90

    # divide into z-slices
    for z_ind in range(nz):
        sp = np.fft.fft2(cube[z_ind])
        sp = np.abs(sp) ** 2  # power spectrum

        sp /= cube.size  # normalize with 1/n (total power)

        sp_s = np.fft.fftshift(sp)  # shift to center

        freq = np.fft.fftfreq(nx, resolution)

        # integrate image azimuthally for frequency shells
        # sp_int = azimuthal_sum(sp_s)
        sp_by_slice[z_ind] = azimuthal_sum(sp_s)

    # smooth
    # sp_int = smooth(sp_int)

    # sum up contribution from each z-slice to obtain integrated spectrum
    sp_int = np.sum(sp_by_slice, axis=0)

    # k vector
    ks = np.arange(1, len(sp_int) + 1, dtype=np.float64)

    # normalize units (for stride also)
    skindepth = 1.0 / resolution
    ks *= 2.0 * np.pi * skindepth / (len(sp_int) + 1)

    # compensate
    # sp_int *= ks**(3.0)
    # sp_int *= 4.0 * np.pi * ks * ks  # 3D 4pi k^2 element
    sp_int *= 2.0 * np.pi * ks  # 2D 2pi k element

    # final proper value inside circle
    max_ks = int(nx // 2)

    # power spectral density
    sp_psd = sp_int[:max_ks]
    ks = ks[:max_ks]
    freq = freq[:max_ks]

    # print("ks", ks)
    # print("fr", freq)

    return ks, sp_psd


def main():
    datadict = parsing()

    field = datadict["field"]

    # L = field.shape[1]  # Number of voxels per side

    # Bz = field[0]
    # By = field[1]
    # Bx = field[2]
    B_vec = np.sqrt(field[0] ** 2 + field[1] ** 2 + field[2] ** 2)

    ks, sp_psd = compute_spec_3d(
        B_vec,
    )

    plt.figure(figsize=(10, 6))
    plt.loglog(ks, sp_psd, marker="o", linestyle="-")
    plt.xlabel(r"$k$", fontsize=12)
    plt.ylabel(r"$\epsilon(k)$", fontsize=12)
    plt.title("Magnetic Power Spectrum", fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
