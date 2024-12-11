import numpy as np
from numba import njit, jit, prange

import yt
from yt import physical_constants as phc
from yt import derived_field
import yt.units as u

import matplotlib.pyplot as plt
import random

random.seed(10)

from athena_read import athdf

from multiprocessing import Pool
from multiprocessing import cpu_count

import argparse, os, pickle

lookup_units = {
    "number_density": "cm**-3",
    "pressure": "Pa",
    "density": "g/cm**3",
    "temperature": "K",
    "normalized_angular_momentum_x": "",
    "normalized_angular_momentum_y": "",
    "normalized_angular_momentum_z": "",
    "vr": "km/s",
    "vtheta": "1/s",
    "vphi": "km/s",
    "vtangent": "km/s",
    "velocity_x": "km/s",
    "velocity_y": "km/s",
    "velocity_z": "km/s",
    "cooling_time": "Myr",
    "sound_crossing_time": "Myr",
    "cooling_rate": "erg/s/cm**3",
    "cooling_ratio": "",
    "normalized_specific_angular_momentum_x": "",
    "normalized_specific_angular_momentum_y": "",
    "normalized_specific_angular_momentum_z": "",
    "free_fall_time": "Myr",
    "free_fall_ratio": "",
    "specific_angular_momentum_x": "km**2/s",
    "specific_angular_momentum_y": "km**2/s",
    "specific_angular_momentum_z": "km**2/s",
    "angular_momentum_x": "km**2/s",
    "angular_momentum_y": "km**2/s",
    "angular_momentum_z": "km**2/s",
    "m_dot_in_cold": "Msun/yr",
    "m_dot_in_hot": "Msun/yr",
    "m_dot_out_cold": "Msun/yr",
    "m_dot_out_hot": "Msun/yr",
}


def regrid(
    file_name: str,
    units_override: dict,
    field=["velocity_x", "velocity_y", "velocity_z", "temperature"],
    dim=256,
    bounding_length=1.0,
):

    # ds = yt.load(file_name,units_override=units_override)
    L = bounding_length

    datacube = np.zeros((dim, dim, dim, len(field)))

    coords_range = np.linspace(-L, L, dim + 1, endpoint=True)
    # mid_coords = [0.5 * (coords_range[i] + coords_range[i+1]) for i in range(dim)]

    # for i_field in range(len(field)):
    print(f"Regridding {field} using {cpu_count()} cores", flush=True)

    with Pool() as p:
        items = [
            (file_name, i, field, dim, coords_range, units_override) for i in range(dim)
        ]

        for i, result in enumerate(p.starmap(find_nearest, items)):
            datacube[i, :, :, :] = result
    return datacube


# Added 11/04/2024: use yt built-in function to regrid
def regrid_yt(
    file_name: str,
    units_override: dict,
    fields=["velocity_x", "velocity_y", "velocity_z", "temperature"],
    dim=[256, 256, 256],
    bounding_length=[1.0, 1.0, 1.0],
    center=[0.0, 0.0, 0.0],
):
    """Regrid an Athena++ hdf5 output into a uniform grid using yt built-in function

    Args:
        file_name (str): file name
        units_override (dict): dict with conversion between physical units and code units
        fields (list[str], optional): the fields being translated to un i. Defaults to ["velocity_x", "velocity_y", "velocity_z", "temperature"].
        dim (list[int], optional): the number of cells in each direction (x, y, z). Defaults to [256, 256, 256].
        bounding_length (list[float], optional): the width of the regrided window in pc in each direction (x, y, z). Defaults to [1., 1., 1.].
        center (list[float], optional): the center of the new grid in pc. Defaults to [0., 0., 0.].
    """

    ds = yt.load(file_name, units_override=units_override)

    out_data = dict()
    # datacube = np.zeros((dim, dim, dim, len(field)))
    bounding_length *= u.kpc
    center *= u.kpc
    x_edges = (center[0] - bounding_length[0] / 2, center[0] + bounding_length[0] / 2)
    y_edges = (center[1] - bounding_length[1] / 2, center[1] + bounding_length[1] / 2)
    z_edges = (center[2] - bounding_length[2] / 2, center[2] + bounding_length[2] / 2)

    gridded_data = ds.r[
        x_edges[0] : x_edges[1] : dim[0] * 1j,
        y_edges[0] : y_edges[1] : dim[1] * 1j,
        z_edges[0] : z_edges[1] : dim[2] * 1j,
    ]
    units_list = [lookup_units[field] for field in fields]

    for (
        i,
        field,
    ) in enumerate(
        fields
    ):  # NOTE: can possibly speed this up with multiprocessing - but a memory bottleneck is possible
        out_data[field] = gridded_data[field].in_units(units_list[i]).value

    return out_data


def find_nearest(
    file_name: str,
    i: int,
    field: list[str],
    dim: int,
    bounds: np.ndarray,
    units_override: dict,
    method="nearest",
):
    ds = yt.load(file_name, units_override=units_override)
    # print("units_override: ", ds.units_override, flush=True)
    print(f"i = {i+1} of {dim} : START", flush=True)
    dataslice = np.zeros((dim, dim, len(field)))

    if method == "interp":
        print("Via interpolation...", flush=True)
        for j in range(dim):
            for k in range(dim):
                region_of_interest = ds.region(
                    center=[0.0, 0.0, 0.0],
                    left_edge=[bounds[i], bounds[j], bounds[k]],
                    right_edge=[bounds[i + 1], bounds[j + 1], bounds[k + 1]],
                )
                for f in range(len(field)):
                    dataslice[j, k, f] = region_of_interest.mean(
                        field[f], weight=("volume")
                    )
    elif method == "nearest":
        print("Via nearest neighbor...", flush=True)
        mid_coords = [0.5 * (bounds[i] + bounds[i + 1]) for i in range(dim)]
        for j in range(dim):
            for k in range(dim):
                for f in range(len(field)):
                    dataslice[j, k, f] = ds.r[
                        mid_coords[i], mid_coords[j], mid_coords[k]
                    ][field[f]]
    else:
        print("Invalid method. Please use 'interp' or 'nearest'", flush=True)
        return None
    print(f"i = {i+1} of {dim} : END", flush=True)
    return dataslice


@njit(parallel=True)
def VSF_3D(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    min_distance: float = 1.0,
    max_distance: float = None,
    n_bins=50,
    order=1,
):
    """Compute first-order velocity structure function (VSF) in 3D, with jit and parallelization

    Args:
        X (np.ndarray): x-coordinates of the data points (pc)
        Y (np.ndarray): y-coordinates of the data points (pc)
        Z (np.ndarray): z-coordinates of the data points (pc)
        vx (np.ndarray): x-components of the velocity vectors (km/s)
        vy (np.ndarray): y-components of the velocity vectors (km/s)
        vz (np.ndarray): z-components of the velocity vectors (km/s)
        max_distance (float, optional): starting distance (pc) in the bins. If None, function assumes 1 pc. Defaults to None.
        max_distance (float, optional): maximum distance (pc) in the bins. If None, function assumes cubic data and take the longest diagonal distance. Defaults to None.
        n_bins (int, optional): number of distance bins. Defaults to 50.
        order (int, optional): order of the VSF. Defaults to 1.

    Returns:
        (np.ndarray, np.ndarray): 1D arrays of the distance bins (pc) and the VSF (km/s) per bin
    """
    # find the maximum distance between any two points in the data
    # assuming the data is roughly of cubic shape (neccessary for the quick computation of max distance)
    if max_distance is None:
        max_distance = np.sqrt(
            (X.max() - X.min()) ** 2
            + (Y.max() - Y.min()) ** 2
            + (Z.max() - Z.min()) ** 2
        )

    if order == 1:
        print("Calculating 1st order VSF")
    elif order == 2:
        print("Calculating 2nd order VSF")
    else:
        print("Order not valid, defaulting to 1st order VSF")
        order = 1

    # print(f"Average supplied x velocity: {np.mean(vx)} km/s")
    # print(f"Average supplied y velocity: {np.mean(vy)} km/s")
    # print(f"Average supplied z velocity: {np.mean(vz)} km/s")

    # print(f"First velocity vector: {vx[0]}, {vy[0]}, {vz[0]} km/s")

    # create bins of equal size in log space
    bins = 10.0 ** np.linspace(np.log10(min_distance), np.log10(max_distance), n_bins)
    squared_bins = bins**2

    vsf_per_bin = np.zeros(n_bins - 1)

    # loop through bins
    for this_bin_index in range(len(squared_bins) - 1):
        if (this_bin_index + 1) % 10 == 0:
            print(f"bin {this_bin_index+1} of {len(squared_bins)-1} : START")
            # print(
            #     f"Distances in this bin: {float(bins[this_bin_index])}-{float(bins[this_bin_index+1])} pc"
            # )
        # for each point in the data, find the distance to all other points, then choose only the distances that are in the same bin
        weights = np.zeros(len(X))
        mean_velocity_differences = np.zeros(len(X))

        for point_a in prange(len(X)):
            squared_distance_to_point_a = (
                (X[point_a] - X) ** 2 + (Y[point_a] - Y) ** 2 + (Z[point_a] - Z) ** 2
            )
            elements_in_this_bin = np.full(len(squared_distance_to_point_a), False)
            elements_in_this_bin[
                (squared_bins[this_bin_index] < squared_distance_to_point_a)
                & (squared_distance_to_point_a <= squared_bins[this_bin_index + 1])
            ] = True
            # don't calculate the same point again
            elements_in_this_bin[:point_a] = False

            squared_velocity_difference_to_point_a = (
                (vx[point_a] - vx[elements_in_this_bin]) ** 2
                + (vy[point_a] - vy[elements_in_this_bin]) ** 2
                + (vz[point_a] - vz[elements_in_this_bin]) ** 2
            )

            # calculate the mean of the velocity differences
            if order == 1:
                if np.sum(elements_in_this_bin) == 0:
                    mean_velocity_differences[point_a] = 0.0
                else:
                    mean_velocity_differences[point_a] = np.mean(
                        np.sqrt(squared_velocity_difference_to_point_a)
                    )
            else:
                mean_velocity_differences[point_a] = np.mean(
                    squared_velocity_difference_to_point_a
                )
            # the number of points in the distance bin is the weight for the mean calculation later
            weights[point_a] = len(squared_velocity_difference_to_point_a)

        if (this_bin_index + 1) % 10 == 0:
            print(f"bin {this_bin_index+1} of {len(squared_bins)-1} : END")
            # print(
            #     f"Mean velocity difference at this bin: {np.mean(mean_velocity_differences)} km/s"
            # )

        # calculate the mean of the velocity differences in this bin
        mean_velocity_differences[weights == 0] = (
            0.0  # set the mean to 0 if there are no points in the bin
        )

        if np.max(weights) == 0:  # if there are no points in the bin, set the VSF to 0
            vsf_per_bin[this_bin_index] = 0.0
        else:
            vsf_per_bin[this_bin_index] = np.average(
                mean_velocity_differences, weights=weights
            )

    return bins, vsf_per_bin


# unit conversions
code_length = 0.064
code_mass = 1.0
code_temperature = 1.0
code_time = 10.0
code_area = code_length**2
code_volume = code_length**3
code_velocity = code_length / code_time
code_momentum = code_velocity * code_mass
code_density = code_mass / code_volume
code_acceleration = code_velocity / code_time
code_force = code_mass * code_acceleration
code_pressure = code_force / code_area
code_energy = code_force * code_length


def makeFilename(pathName: str, baseExtension: str, n: int) -> str:
    if n < 10:
        file_n = "0000" + str(n)
    elif (n >= 10) & (n < 100):
        file_n = "000" + str(n)
    elif n >= 1000:
        file_n = "0" + str(n)
    else:
        file_n = "00" + str(n)

    return f"{pathName}{baseExtension}{file_n}.athdf"


mu = 0.5924489
kpcCGS = 3.08567758096e21
MpcCGS = kpcCGS * 1.0e3
kmCGS = 1.0e5
yearCGS = 31557600.0
MyrCGS = yearCGS * 1.0e6
solarMassCGS = 1.988e33
boltzmannConstCGS = 1.3806488e-16
hydrogenMassCGS = 1.6735575e-24

boltzmannConstAstronomical = boltzmannConstCGS / (solarMassCGS * MpcCGS**2 * MyrCGS**-2)
codeBoltzmannConst = boltzmannConstAstronomical / (code_energy / code_temperature)
kms_Astronomical = kmCGS / (MpcCGS / MyrCGS)
hydrogenMassAstronomical = hydrogenMassCGS / solarMassCGS

SMBHMass = 6.5e9  # solar masses

# logTemperatureArray = np.array([3.8, 3.84, 3.88, 3.92, 3.96, 4., 4.04, 4.08, 4.12, 4.16, 4.2,
#                                           4.24, 4.28, 4.32, 4.36, 4.4, 4.44, 4.48, 4.52, 4.56, 4.6, 4.64,
#                                           4.68, 4.72, 4.76, 4.8, 4.84, 4.88, 4.92, 4.96, 5., 5.04, 5.08,
#                                           5.12, 5.16, 5.2, 5.24, 5.28, 5.32, 5.36, 5.4, 5.44, 5.48, 5.52,
#                                           5.56, 5.6, 5.64, 5.68, 5.72, 5.76, 5.8, 5.84, 5.88, 5.92, 5.96,
#                                           6., 6.04, 6.08, 6.12, 6.16, 6.2, 6.24, 6.28, 6.32, 6.36, 6.4,
#                                           6.44, 6.48, 6.52, 6.56, 6.6, 6.64, 6.68, 6.72, 6.76, 6.8, 6.84,
#                                           6.88, 6.92, 6.96, 7., 7.04, 7.08, 7.12, 7.16, 7.2, 7.24, 7.28,
#                                           7.32, 7.36, 7.4, 7.44, 7.48, 7.52, 7.56, 7.6, 7.64, 7.68, 7.72,
#                                           7.76, 7.8, 7.84, 7.88, 7.92, 7.96, 8., 8.04, 8.08, 8.12, 8.16])
# logEmissivityHydroArray = np.array([-30.6104, -29.4107, -28.4601, -27.5743, -26.3766, -25.289,
#                                               -24.2684, -23.3834, -22.5977, -21.9689, -21.5972, -21.4615,
#                                               -21.4789, -21.5497, -21.6211, -21.6595, -21.6426, -21.5688,
#                                               -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
#                                               -20.8986, -20.8281, -20.77, -20.7223, -20.6888, -20.6739,
#                                               -20.6815, -20.7051, -20.7229, -20.7208, -20.7058, -20.6896,
#                                               -20.6797, -20.6749, -20.6709, -20.6748, -20.7089, -20.8031,
#                                               -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
#                                               -21.4538, -21.5055, -21.574, -21.63, -21.6615, -21.6766,
#                                               -21.6886, -21.7073, -21.7304, -21.7491, -21.7607, -21.7701,
#                                               -21.7877, -21.8243, -21.8875, -21.9738, -22.0671, -22.1537,
#                                               -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
#                                               -22.359, -22.3512, -22.342, -22.3342, -22.3312, -22.3346,
#                                               -22.3445, -22.3595, -22.378, -22.4007, -22.4289, -22.4625,
#                                               -22.4995, -22.5353, -22.5659, -22.5895, -22.6059, -22.6161,
#                                               -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
#                                               -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.514,
#                                               -22.4992, -22.4844, -22.4695, -22.4543, -22.4392, -22.4237,
#                                               -22.4087, -22.3928])

# tempList = np.logspace(3.,9.,200)

# def emissivityFromTemperature(temperature):
#     #Real emissivityCGS, emissivityAstronomical;
#     logTemperature = np.log10(temperature)
#     if logTemperature <= 4.2: # Koyama & Inutsuka (2002)
#         emissivityCGS = (2.0e-19 * np.exp(-1.184e5 / (temperature + 1.e3)) + 2.8e-28 * np.sqrt(temperature) * np.exp(-92. / temperature))
#     elif logTemperature > 8.15: # Schneider & Robertson (2018)
#         emissivityCGS = 10.**(0.45 * logTemperature - 26.065)
#     else: # Schure+09
#         emissivityCGS = 10.**(np.interp(logTemperature, logTemperatureArray, logEmissivityHydroArray))

#     # emissivityAstronomical = emissivityCGS / (solarMassCGS * pow(MpcCGS, 5) * pow(MyrCGS, -3))

#     return emissivityCGS

# lambdaListCGS = np.zeros(200)
# for i, temp in enumerate(tempList):
#     lambdaListCGS[i] = emissivityFromTemperature(temp)

# @derived_field(name="cooling_time", sampling_type="cell", units="Myr", force_override=True)
# def _cooling_time(field, data):
#     return ( (5/2) *  phc.kboltz * (data["gas", "temperature"].to("K")) / ((data["gas", "number_density"].to("cm**-3")) * np.interp(data["gas", "temperature"].to("K"), tempList, lambdaListCGS) * (ds.units.erg / ds.units.s * ds.units.cm**3)))

# @derived_field(name="sound_crossing_time", sampling_type="cell", units="Myr", force_override=True)
# def _sound_crossing_time(field, data):
#     return data["gas", "dx"] / np.sqrt(data.ds.gamma * data["gas", "pressure"] / data["gas", "density"])

# @derived_field(name="cooling_ratio", sampling_type="cell", force_override=True)
# def _cooling_ratio(field, data):
#     return data["gas", "cooling_time"] / data["gas", "sound_crossing_time"]

# @derived_field(name="keplerian_speed", sampling_type="cell", units="Mpc/Myr", force_override=True)
# def _keplerian_speed(field, data):
#     return np.sqrt(phc.G * SMBHMass * phc.msun / data["index", "radius"])

# @derived_field(name="keplerian_specific_angular_momentum", sampling_type="cell", units="Mpc**2/Myr", force_override=True)
# def _keplerian_specific_angular_momentum(field, data):
#     return data["gas", "keplerian_speed"] * data["index", "radius"]

# @derived_field(name="normalized_specific_angular_momentum_x", sampling_type="cell", force_override=True)
# def _normalized_specific_angular_momentum_x(field, data):
#     return data["gas", "specific_angular_momentum_x"] / data["gas", "keplerian_specific_angular_momentum"]

# @derived_field(name="normalized_specific_angular_momentum_y", sampling_type="cell", force_override=True)
# def _normalized_specific_angular_momentum_y(field, data):
#     return data["gas", "specific_angular_momentum_y"] / data["gas", "keplerian_specific_angular_momentum"]

# @derived_field(name="normalized_specific_angular_momentum_z", sampling_type="cell", force_override=True)
# def _normalized_specific_angular_momentum_z(field, data):
#     return data["gas", "specific_angular_momentum_z"] / data["gas", "keplerian_specific_angular_momentum"]

# @derived_field(name="keplerian_angular_momentum", sampling_type="cell", force_override=True)
# def _keplerian_angular_momentum(field, data):
#     return data["gas", "mass"] * data["gas", "keplerian_specific_angular_momentum"]

# @derived_field(name="normalized_angular_momentum_x", sampling_type="cell", force_override=True)
# def _normalized_angular_momentum_x(field, data):
#     return data["gas", "angular_momentum_x"] / data["gas", "keplerian_angular_momentum"]

# @derived_field(name="normalized_angular_momentum_y", sampling_type="cell", force_override=True)
# def _normalized_angular_momentum_y(field, data):
#     return data["gas", "angular_momentum_y"] / data["gas", "keplerian_angular_momentum"]

# @derived_field(name="normalized_angular_momentum_z", sampling_type="cell", force_override=True)
# def _normalized_angular_momentum_z(field, data):
#     return data["gas", "angular_momentum_z"] / data["gas", "keplerian_angular_momentum"]

# @derived_field(name="free_fall_time", sampling_type="cell", units="Myr", force_override=True)
# def _free_fall_time(field, data):
#     return np.pi / 2 * data["index","radius"]**(3/2) / np.sqrt(2 * phc.G * (SMBHMass * phc.msun + data["gas", "mass"]))

# @derived_field(name="free_fall_ratio", sampling_type="cell", force_override=True)
# def _free_fall_ratio(field, data):
#     return data["gas", "cooling_time"] / data["gas", "free_fall_time"]

# @derived_field(name="density_squared", sampling_type="cell", units="msun**2*Mpc**-6", force_override=True)
# def _density_squared(field, data):
#     return data["gas", "density"]**2


@derived_field(name="angle_theta", sampling_type="cell", force_override=True)
def _angle_theta(field, data):
    return np.arccos(data["index", "z"] / data["index", "radius"])


if __name__ == "__main__":
    units_override = {
        "length_unit": (code_length, "Mpc"),
        "time_unit": (code_time, "Myr"),
        "mass_unit": (code_mass, "Msun"),
        "temperature_unit": (code_temperature, "K"),
    }

    parser = argparse.ArgumentParser(description="VSF calculation")
    parser.add_argument(
        "--path",
        type=str,
        default=os.getcwd(),
        help="path to the simulation data",
        required=False,
    )
    parser.add_argument(
        "--base_ext", type=str, help="base extension of the simulation data"
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        nargs="+",
        action="store",
        help="range of snapshots (start, end)",
    )
    parser.add_argument("--size", type=float, help="window size for the plots in kpc")
    parser.add_argument(
        "--dim", type=int, default=128, help="number of cells in each direction"
    )
    parser.add_argument(
        "--cgm_cut",
        type=float,
        default=5.0e-25,
        help="density cut for the CGM in g/cm^3",
    )
    args = parser.parse_args()

    f_kin = 0.8
    # path = f'/mnt/home/tha10/ceph/M87/galaxy-scale/product-{f_kin}/'
    snapshots = (args.snapshots[0], args.snapshots[1])
    path = args.path + "/"
    base_ext = args.base_ext

    window_size = args.size * u.kpc

    grid_size = args.dim
    grid_resolution = window_size / grid_size

    cgm_density_cut = args.cgm_cut * u.g / u.cm**3

    print(f"Path: {path}", flush=True)
    print(f"Base extension: {base_ext}", flush=True)
    print(f"Snapshots: {snapshots}", flush=True)
    print(f"Window size: {window_size}", flush=True)
    print(f"Grid size: {grid_size}", flush=True)
    print(f"Grid resolution: {grid_resolution}", flush=True)
    print(f"CGM density cut: {cgm_density_cut}", flush=True)

    for n in range(snapshots[0], snapshots[1] + 1):
        myfilename = makeFilename(path, base_ext, n=n)
        ds = yt.load(myfilename, units_override=units_override)
        time = int(ds.current_time.to("kyr"))
        print(f"Simulation time: {time} kyr", flush=True)

        # regridding data with athena built-in function
        if False:
            box_width = 1 * 1.0e-3  # Mpc
            bounding_length = box_width / 2 / code_length
            datadict = athdf(
                myfilename,
                quantities=("vel1", "vel2", "vel3", "press", "rho"),
                level=9,
                subsample=True,
                x1_min=-bounding_length,
                x1_max=bounding_length,
                x2_min=-bounding_length,
                x2_max=bounding_length,
                x3_min=-bounding_length,
                x3_max=bounding_length,
            )

            print("Regridding done", flush=True)
            # print(datadict.keys(), flush=True)
            print(datadict["vel1"].shape, flush=True)

            datacube = np.empty(
                (
                    datadict["vel1"].shape[0],
                    datadict["vel1"].shape[1],
                    datadict["vel1"].shape[2],
                    5,
                ),
                dtype=np.float64,
            )
            datacube[:, :, :, 0] = datadict["vel3"]
            datacube[:, :, :, 1] = datadict["vel2"]
            datacube[:, :, :, 2] = datadict["vel1"]
            datacube[:, :, :, 3] = datadict["press"]
            datacube[:, :, :, 4] = datadict["rho"]

            # convert these into physical units and calculate the temperature
            velocities = (
                datacube[:, :, :, 0:3] / kms_Astronomical * code_velocity
            )  # in km/s
            velocity_magnitude = np.sqrt(
                velocities[:, :, :, 0] ** 2
                + velocities[:, :, :, 1] ** 2
                + velocities[:, :, :, 2] ** 2
            )
            print(
                "Velocity range (km/s): ",
                np.min(velocity_magnitude),
                np.max(velocity_magnitude),
                flush=True,
            )

            # calculate temperature
            # print("Density range (code units): ", np.min(datacube[:,:,:,4]), np.max(datacube[:,:,:,4]), flush=True)
            # print("conversion: ", mu * hydrogenMassAstronomical, flush=True)
            number_density_code = datacube[:, :, :, 4] / (mu * hydrogenMassAstronomical)
            print(
                "Number density range (cm**-3): ",
                np.min(number_density_code) * (code_volume**3),
                np.max(number_density_code) * (code_volume**3),
                flush=True,
            )
            temperature = (
                datacube[:, :, :, 3]
                / (number_density_code * codeBoltzmannConst)
                * code_temperature
            )
            print(
                "Temperature range (K): ",
                np.min(temperature),
                np.max(temperature),
                flush=True,
            )

            if False:
                save_name = f"regridded_prim_{time}Myr_{box_width}Mpc.npy"
                np.save(path + save_name, datacube)
                print(f"Regridded raw data saved in {path} as {save_name}", flush=True)

            if False:
                save_name1 = f"regridded_vel_{time}Myr_{box_width}Mpc.npy"
                np.save(path + save_name1, velocities)
                save_name2 = f"regridded_temperature_{time}Myr_{box_width}Mpc.npy"
                np.save(path + save_name2, temperature)
                print(
                    f"Regridded processed data saved in {path} as {save_name1} and {save_name2}",
                    flush=True,
                )

        # regridding data onto a uniform grid
        if True:

            # dim = 256  # 15m for 128^3 on 1 Rome node, 2h20m for 256^3 (for one field)
            datadict = regrid_yt(
                myfilename,
                units_override,
                fields=["velocity_x", "velocity_y", "velocity_z", "density"],
                dim=[grid_size, grid_size, grid_size],
                bounding_length=[
                    window_size.value,
                    window_size.value,
                    window_size.value,
                ],
            )
            print("Regridding done", flush=True)
            print(datadict.keys(), flush=True)

            save_to_file = False

            if save_to_file:
                save_name = f"regridded_data_{time}kyr_{window_size.value}kpc.pkl"
                with open(path + save_name, "wb") as f:
                    pickle.dump(datadict, f)
                print(f"Regridded data saved in {path} as {save_name}", flush=True)

        # VSF calculation
        if True:
            read_data_from_file = False
            # get uniform grid of positions and import velocity and temperature data
            if read_data_from_file:
                dim = 256
                box_width = 10 * 1.0e-3  # Mpc

                position_array = np.linspace(
                    -box_width * 1.0e3 / 2,
                    box_width * 1.0e3 / 2,
                    dim + 1,
                    endpoint=True,
                )
                mid_points = [
                    0.5 * (position_array[i] + position_array[i + 1])
                    for i in range(dim)
                ]
                mid_points = np.array(mid_points) * 1.0e3

                x_pos, y_pos, z_pos = np.meshgrid(
                    mid_points, mid_points, mid_points, indexing="ij"
                )

                datacube = np.load(
                    path + f"regridded_data_{time}kyr_{box_width}kpc.npy"
                )
                temperature = (
                    datacube[:, :, :, 3] * code_temperature
                )  # convert from code units to K
                velocity_datacube = (
                    datacube[:, :, :, :3] / kms_Astronomical * code_velocity
                )  # convert from code units to km/s
                Vx = velocity_datacube[:, :, :, 0]
                Vy = velocity_datacube[:, :, :, 1]
                Vz = velocity_datacube[:, :, :, 2]
                V_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2)

            if True:
                position_array = (
                    np.linspace(
                        -window_size / 2,
                        window_size / 2,
                        grid_size + 1,
                        endpoint=True,
                    )
                    .in_units("pc")
                    .value
                )
                # print("Position array: ", position_array, flush=True)
                mid_points = [
                    0.5 * (position_array[i] + position_array[i + 1])
                    for i in range(grid_size)
                ]  # the locations of the center of the cells
                mid_points = np.array(mid_points)

                z_pos, y_pos, x_pos = np.meshgrid(
                    mid_points,
                    mid_points,
                    mid_points,
                    indexing="ij",
                )

                Vz = datadict["velocity_z"]
                Vy = datadict["velocity_y"]
                Vx = datadict["velocity_x"]
                density = datadict["density"]
                cgm_mask = np.where(density < cgm_density_cut.value, True, False)
                # V_mag = velocity_magnitude
                print("Number of cells in total: ", grid_size**3, flush=True)

            separate_temperature = False
            if separate_temperature:
                # mask data to separate cold and hot gas
                cold_gas_mask = np.where(temperature < 8.0e5, True, False)
                hot_gas_mask = np.where(temperature > 6.0e6, True, False)

                # eliminate all cells with high velocity (along the jet)
                velocity_cut = 1.0e3  # km/s

                Vmag_mask = np.where(V_mag < velocity_cut, True, False)
                # print("Number of cells in Vmag mask: ", np.sum(Vmag_mask), flush=True)

                # final masks
                cold_gas_mask = np.logical_and(cold_gas_mask, Vmag_mask)
                hot_gas_mask = np.logical_and(hot_gas_mask, Vmag_mask)

                # take positions and velocities of cold gas
                X = x_pos[cold_gas_mask]
                Y = y_pos[cold_gas_mask]
                Z = z_pos[cold_gas_mask]
                vx = Vx[cold_gas_mask]
                vy = Vy[cold_gas_mask]
                vz = Vz[cold_gas_mask]
                print("Number of cells in cold gas: ", len(X), flush=True)

                # calculate VSF of cold gas
                random.seed(42)
                sample_size = int(1.0e5)

                if len(X) > sample_size:
                    print(f"Sampling cold gas as {sample_size} points")
                    random_indices = random.sample(range(len(X)), sample_size)
                    X = X[random_indices]
                    Y = Y[random_indices]
                    Z = Z[random_indices]
                    vx = vx[random_indices]
                    vy = vy[random_indices]
                    vz = vz[random_indices]

                n_bins = 41
                min_distance = 2.5
                max_distance = 2.0e3

                dist_array_cold_b, v_diff_mean_cold_b = VSF_3D(
                    X,
                    Y,
                    Z,
                    vx,
                    vy,
                    vz,
                    min_distance=min_distance,
                    max_distance=max_distance,
                    n_bins=n_bins,
                )

                print("Cold gas VSF done", flush=True)

                # take positions and velocities of hot gas
                X = x_pos[hot_gas_mask]
                Y = y_pos[hot_gas_mask]
                Z = z_pos[hot_gas_mask]
                vx = Vx[hot_gas_mask]
                vy = Vy[hot_gas_mask]
                vz = Vz[hot_gas_mask]
                print("Number of cells in hot gas: ", len(X), flush=True)

                random.seed(41)
                if len(X) > sample_size:
                    print(f"Sampling hot gas as {sample_size} points")
                    random_indices = random.sample(range(len(X)), sample_size)
                    X = X[random_indices]
                    Y = Y[random_indices]
                    Z = Z[random_indices]
                    vx = vx[random_indices]
                    vy = vy[random_indices]
                    vz = vz[random_indices]

                # calculate VSF of hot gas
                n_bins = 41
                min_distance = 2.5
                max_distance = 2.0e3

                dist_array_hot, v_diff_mean_hot = VSF_3D(
                    X,
                    Y,
                    Z,
                    vx,
                    vy,
                    vz,
                    min_distance=min_distance,
                    max_distance=max_distance,
                    n_bins=n_bins,
                )

                plt.figure(figsize=(10, 8), dpi=300)
                plt.loglog(
                    dist_array_cold_b[:-2] * 1.0e-3,
                    v_diff_mean_cold_b[:-1],
                    linewidth=2,
                    c="C0",
                    label="Cold",
                    marker="o",
                    markersize=7,
                    linestyle="-",
                )
                plt.loglog(
                    dist_array_hot[:-2] * 1.0e-3,
                    v_diff_mean_hot[:-1],
                    linewidth=2,
                    c="C1",
                    label="Hot",
                    marker="o",
                    markersize=7,
                    linestyle="-",
                )

                full_ell_range = np.logspace(
                    np.log10(min_distance * 2 * 1.0e-3),
                    np.log10(max_distance / 2 * 1.0e-3),
                    50,
                )
                ell_1_2 = full_ell_range ** (0.5) * 6.0e2
                plt.plot(full_ell_range, ell_1_2, linestyle="--", c="C3", linewidth=2)
                # plt.text(5.e-2, 1.2e2, r'$\ell^{0.58}$', fontsize=18)

                ell_1_3 = full_ell_range ** (1.0 / 3.0) * 9.0e2
                plt.plot(full_ell_range, ell_1_3, linestyle="-.", c="C6", linewidth=2)
                # plt.text(2.7e-1, 5.e2, r'$\ell^{0.8}$', fontsize=18)h=2)

                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                # 10 kpc box
                # plt.xlim(1.8e-2, 2.e1)
                # plt.ylim(8.e-1, 2.2e3)

                # 1 kpc box

                plt.ylim(9.0e-1, 1.4e3)
                plt.xlabel(r"$\ell$ (kpc)", fontsize=22)
                plt.ylabel(
                    r"$\langle \delta \mathbf{v} \rangle$ (km s$^{-1}$)", fontsize=22
                )
                plt.title(f"VSF at {time} Myr, f = {f_kin}", fontsize=22)
                plt.grid()
                plt.legend(fontsize=20)

                path += "vsf-1kpc/"
                if n < 10:
                    file_n = "0000" + str(n)
                elif (n >= 10) & (n < 100):
                    file_n = "000" + str(n)
                else:
                    file_n = "00" + str(n)
                plt.savefig(path + f"VSF_{file_n}.png")

                print("Figure saved to ", path, flush=True)

            else:
                X = x_pos[cgm_mask]
                Y = y_pos[cgm_mask]
                Z = z_pos[cgm_mask]
                vx = Vx[cgm_mask]
                vy = Vy[cgm_mask]
                vz = Vz[cgm_mask]
                print("Number of cells in the CGM: ", len(X), flush=True)

                random.seed(42)
                sample_size = int(1.0e4)

                if len(X) > sample_size:
                    print(f"Sampling CGM as {sample_size} points")
                    random_indices = random.sample(range(len(X)), sample_size)
                    X = X[random_indices]
                    Y = Y[random_indices]
                    Z = Z[random_indices]
                    vx = vx[random_indices]
                    vy = vy[random_indices]
                    vz = vz[random_indices]
                else:
                    sample_size = len(X)

                # V_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                # print(
                #     "Velocity range (km/s): ", np.min(V_mag), np.max(V_mag), flush=True
                # )

                n_bins = 50
                min_distance = grid_resolution.in_units("pc").value * 4
                max_distance = window_size.in_units("pc").value

                dist_array, v_diff_mean = VSF_3D(
                    X,
                    Y,
                    Z,
                    vx,
                    vy,
                    vz,
                    min_distance=min_distance,
                    # max_distance=max_distance,
                    n_bins=n_bins,
                )
                print("distance array: ", dist_array, flush=True)
                print("v_diff_mean: ", v_diff_mean, flush=True)

                plt.figure(figsize=(10, 8), dpi=300)
                plt.plot(
                    dist_array[:-2] * 1.0e-3,
                    v_diff_mean[:-1],
                    linewidth=2,
                    c="C0",
                    label="CGM",
                    marker="o",
                    markersize=7,
                    linestyle="-",
                )
                # plt.xticks(fontsize=20)
                # plt.yticks(fontsize=20)
                plt.show()
