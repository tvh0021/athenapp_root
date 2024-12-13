import numpy as np
from numba import njit, prange

import yt
import yt.units as u

import matplotlib.pyplot as plt
import random

random.seed(10)

# from athena_read import athdf

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
    time = int(ds.current_time.to("kyr").value)
    print(f"Simulation time: {time} kyr", flush=True)

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

    return out_data, time


# Things I have tried (unsuccesfully) to speed up the VSF calculation: using KD-trees to "cache" the distances between points.
# Use joblib to parallelize the VSF calculation. All either runs out of memory or takes longer than the current implementation.
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
    # assuming the data is roughly of cubic shape (necessary for the quick computation of max distance)
    if max_distance is None:
        max_distance = np.sqrt(
            (X.max() - X.min()) ** 2
            + (Y.max() - Y.min()) ** 2
            + (Z.max() - Z.min()) ** 2
        )

    if order == 1:
        print("Calculating 1st order VSF")
        print("Number of points: ", len(X))
    elif order == 2:
        print("Calculating 2nd order VSF")
        print("Number of points: ", len(X))
    else:
        print("Order not valid, defaulting to 1st order VSF")
        order = 1

    # create bins of equal size in log space
    bins = 10.0 ** np.linspace(np.log10(min_distance), np.log10(max_distance), n_bins)
    squared_bins = bins**2

    vsf_per_bin = np.zeros(n_bins - 1)

    # loop through bins
    for this_bin_index in prange(len(squared_bins) - 1):
        if (this_bin_index) % 10 == 0:
            print(f"bin {this_bin_index+1} of {len(squared_bins)} : START")
            # print(
            #     f"Distances in this bin: {float(bins[this_bin_index])}-{float(bins[this_bin_index+1])} pc"
            # )
        # for each point in the data, find the distance to all other points, then choose only the distances that are in the same bin
        bin_vel_sum = 0.0
        bin_count = 0

        bin_lower = squared_bins[this_bin_index]
        bin_upper = squared_bins[this_bin_index + 1]

        for point_a in range(len(X)):

            dx = X[point_a] - X
            dy = Y[point_a] - Y
            dz = Z[point_a] - Z
            squared_distance_to_point_a = dx**2 + dy**2 + dz**2

            mask = (bin_lower < squared_distance_to_point_a) & (
                squared_distance_to_point_a <= bin_upper
            )
            mask[:point_a] = False  # don't calculate the same point again

            if np.any(mask):
                dvx = vx[point_a] - vx[mask]
                dvy = vy[point_a] - vy[mask]
                dvz = vz[point_a] - vz[mask]
                squared_velocity_difference_to_point_a = dvx**2 + dvy**2 + dvz**2

                # compute VSF
                if order == 1:
                    bin_vel_sum += np.sum(
                        np.sqrt(squared_velocity_difference_to_point_a)
                    )
                else:
                    bin_vel_sum += np.sum(squared_velocity_difference_to_point_a)

                bin_count += np.sum(mask)
        if (this_bin_index) % 10 == 0:
            print(f"bin {this_bin_index+1} of {len(squared_bins)} : END")

        if bin_count > 0:
            vsf_per_bin[this_bin_index] = bin_vel_sum / bin_count
        else:
            vsf_per_bin[this_bin_index] = 0.0

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

SMBHMass = 6.5e9 * u.Msun  # solar masses


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
    parser.add_argument(
        "--vsf_order",
        type=int,
        default=1,
        help="order of the VSF calculation (1 or 2)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="number of bins for the VSF calculation",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="number of points to sample for the VSF calculation",
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
        # ds = yt.load(myfilename, units_override=units_override)

        datadict, time = regrid_yt(
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
            if n < 10:
                file_n = "0000" + str(n)
            elif (n >= 10) & (n < 100):
                file_n = "000" + str(n)
            elif n >= 1000:
                file_n = "0" + str(n)
            else:
                file_n = "00" + str(n)
            save_name = f"regridded_data_{file_n}_{window_size.value}kpc.pkl"
            with open(path + save_name, "wb") as f:
                pickle.dump(datadict, f)
            print(f"Regridded data saved in {path} as {save_name}", flush=True)

        # VSF calculation
        calculate_vsf = True
        if calculate_vsf:
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

            # separate the data into cold and hot gas; this doesn't matter when we only care about the CGM
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
                    # max_distance=max_distance,
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
                    r"$\langle \delta \mathbf{v} \rangle$ (km s$^{-1}$)",
                    fontsize=22,
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

                # mask data to separate CGM
                sixd_cgm = np.zeros((int(np.sum(cgm_mask)), 6), dtype=np.float64)
                sixd_cgm[:, 0] = x_pos[cgm_mask]
                sixd_cgm[:, 1] = y_pos[cgm_mask]
                sixd_cgm[:, 2] = z_pos[cgm_mask]
                sixd_cgm[:, 3] = Vx[cgm_mask]
                sixd_cgm[:, 4] = Vy[cgm_mask]
                sixd_cgm[:, 5] = Vz[cgm_mask]

                print("Number of cells in the CGM: ", sixd_cgm.shape[0], flush=True)

                sample_size = args.sample_size
                if sample_size is None:
                    sample_size = sixd_cgm.shape[0]
                    sixd_sample = sixd_cgm
                else:
                    sample_size = int(sample_size)

                    print(f"Sampling CGM as {sample_size} points")
                    random_indices = np.random.choice(
                        sixd_cgm.shape[0], sample_size, replace=False
                    )
                    sixd_sample = sixd_cgm[random_indices, :]

                # n_bins = args.n_bins
                min_distance = grid_resolution.in_units("pc").value * 4
                max_distance = float(window_size.in_units("pc").value) * np.sqrt(2)
                print("Starting VSF calculation", flush=True)

                dist_array, v_diff_mean = VSF_3D(
                    sixd_sample[:, 0],
                    sixd_sample[:, 1],
                    sixd_sample[:, 2],
                    sixd_sample[:, 3],
                    sixd_sample[:, 4],
                    sixd_sample[:, 5],
                    min_distance=min_distance,
                    max_distance=max_distance,
                    n_bins=args.n_bins,
                    order=args.vsf_order,
                )

                print("distance array: ", dist_array, flush=True)
                print("v_diff_mean: ", v_diff_mean, flush=True)

                plt.figure(figsize=(10, 8), dpi=300)
                plt.loglog(
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
                # plt.show()

                full_ell_range = np.logspace(
                    np.log10(min_distance * 2 * 1.0e-3),
                    np.log10(max_distance / 2 * 1.0e-3),
                    50,
                )
                ell_1_2 = full_ell_range ** (0.5) * 1.0e2
                plt.plot(
                    full_ell_range,
                    ell_1_2,
                    linestyle="--",
                    c="C3",
                    linewidth=2,
                    label=r"$\ell^{1/2}$",
                )
                # plt.text(5.e-2, 1.2e2, r'$\ell^{0.58}$', fontsize=18)

                ell_1_3 = full_ell_range ** (1.0 / 3.0) * 1.0e2
                plt.plot(
                    full_ell_range,
                    ell_1_3,
                    linestyle="-.",
                    c="C6",
                    linewidth=2,
                    label=r"$\ell^{1/3}$",
                )
                # plt.text(2.7e-1, 5.e2, r'$\ell^{0.8}$', fontsize=18)h=2)

                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                # 10 kpc box
                # plt.xlim(1.8e-2, 2.e1)
                # plt.ylim(8.e-1, 2.2e3)

                # 1 kpc box

                plt.ylim(v_diff_mean.min() / 1.5, v_diff_mean.max() * 1.2)
                plt.xlabel(r"$\ell$ (kpc)", fontsize=22)
                plt.ylabel(
                    r"$\langle \delta \mathbf{v} \rangle$ (km s$^{-1}$)",
                    fontsize=22,
                )
                plt.title(f"VSF at {time} kyr, f = {f_kin}", fontsize=22)
                plt.grid()
                plt.legend(fontsize=20)

                if n < 10:
                    file_n = "0000" + str(n)
                elif (n >= 10) & (n < 100):
                    file_n = "000" + str(n)
                elif n >= 1000:
                    file_n = "0" + str(n)
                else:
                    file_n = "00" + str(n)

                name = (
                    f"VSF_{file_n}_{window_size.value}_d{grid_size}_s{sample_size}.png"
                )
                plt.savefig(path + name)

                print(f"Figure saved to {path} as {name}", flush=True)
