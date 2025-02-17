### Script to decompose different components of the pressure support in the CGM
### following Lochaas+23, by Trung Ha

import athena_parameters as ap

import numpy as np

# from numba import njit, prange, set_num_threads

import yt
import yt.units as u
from yt import physical_constants as phc

from scipy.ndimage import gaussian_filter

# from scipy.ndimage import convolve

import matplotlib.pyplot as plt

import argparse, os, pickle, gc


def get_data(path, base_ext, snapshot):
    """Load the data from the simulation"""
    my_file_name = ap.makeFilename(path, base_ext, snapshot)
    return yt.load(my_file_name, units_override=ap.units_override)


def smooth_field(data, sigma):
    """Smooth the data with a Gaussian filter"""
    gauss = gaussian_filter(data, sigma=sigma)
    return gauss


def compute_turbulent_velocity_squared(v_vec, v_smooth_vec):
    """Compute the turbulent velocity squared from the velocity field and the smoothed velocity field
    see Eq. 4 in Lochaas+23

    Args:
        v_vec (np.ndarray): 4d array of the velocity field (n, i, j, k) in km/s
        v_smooth_vec (np.ndarray): 4d array of the smoothed velocity field (n, i, j, k) in km/s

    Returns:
        np.ndarray: 3d array of the turbulent velocity squared (i, j, k) in km^2/s^2
    """
    return (
        1.0
        / 3.0
        * np.einsum("nijk,nijk->ijk", v_vec - v_smooth_vec, v_vec - v_smooth_vec)
        * (u.km / u.s) ** 2
    )


def compute_turbulent_pressure(v_vec, v_smooth_vec, density_smooth):
    """Eq. 5 in Lochaas+23"""
    return density_smooth * compute_turbulent_velocity_squared(v_vec, v_smooth_vec)


def compute_ram_pressure(vr_smooth_dr, density_smooth, grid_resolution):
    """Eq. 6 in Lochaas+23"""
    return density_smooth * (vr_smooth_dr * grid_resolution) ** 2


def compute_rotational_force(vtheta_smooth, vphi_smooth, r):
    """Eq. 7 in Lochaas+23"""
    return (vtheta_smooth**2 + vphi_smooth**2) / r


def compute_force_from_pressure(pressure, density, grid_resolution):
    """Compute the force magnitude from the pressure field

    Args:
        pressure (np.ndarray): 3d array of the pressure field in Pa
        density (np.ndarray): 3d array of the density field in g/cm^3
        grid_resolution (float): grid resolution in kpc

        Returns:
        np.ndarray: 3d array of the force magnitude in g/cm/s^2

    """
    force_vector = np.asarray(np.gradient(pressure, grid_resolution, axis=(0, 1, 2)))
    return (
        np.einsum("nijk,nijk->ijk", force_vector, force_vector)
        * (u.Pa / u.kpc)
        / density
    )


if __name__ == "__main__":
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
        "--l0", type=float, default=10, help="turbulence driving scale in kpc"
    )
    parser.add_argument(
        "--cgm_cut",
        type=float,
        default=5.0e-25,
        help="density cut for the CGM in g/cm^3",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="number of points to sample for the VSF calculation",
    )
    args = parser.parse_args()

    f_kin = 0.8
    snapshots = (args.snapshots[0], args.snapshots[1])
    path = args.path + "/"
    base_ext = args.base_ext

    window_size = args.size * u.kpc

    grid_size = args.dim
    grid_resolution = window_size / grid_size
    driving_scale = args.l0 * u.kpc

    cgm_density_cut = args.cgm_cut * u.g / u.cm**3

    print(f"Path: {path}", flush=True)
    print(f"Base extension: {base_ext}", flush=True)
    print(f"Snapshots: {snapshots}", flush=True)
    print(f"Window size: {window_size}", flush=True)
    print(f"Grid size: {grid_size}", flush=True)
    print(f"Grid resolution: {grid_resolution}", flush=True)
    print(f"Turbulence driving scale: {driving_scale}", flush=True)
    print(f"CGM density cut: {cgm_density_cut}", flush=True)

    for snapshot in range(snapshots[0], snapshots[1] + 1):

        # Load yt object
        ds = get_data(path, base_ext, snapshot)

        # Regrid the data into a uniform grid
        datadict, time = ap.regrid_yt(
            ds,
            fields=[
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "pressure_gradient_x",
                "pressure_gradient_y",
                "pressure_gradient_z",
                "density",
                "vr",
                "vtheta",
                "vphi",
                "radius",
            ],
            dim=[grid_size, grid_size, grid_size],
            bounding_length=[
                window_size.value,
                window_size.value,
                window_size.value,
            ],
        )

        print(f"Snapshot: {snapshot}, time: {time}", flush=True)
        # Clean up the yt object
        del ds
        gc.collect()

        ### Compute the force from thermal pressure
        force_thermal = (
            1.0
            / datadict["density"]
            * np.sqrt(
                datadict["pressure_gradient_x"].value ** 2
                + datadict["pressure_gradient_y"].value ** 2
                + datadict["pressure_gradient_z"].value ** 2
            ).in_units("Pa/cm")
        )

        # Compute the smoothed velocity field
        sigma = driving_scale / grid_resolution / 6.0

        datadict["velocity_smooth_x"] = smooth_field(datadict["velocity_x"], sigma)
        datadict["velocity_smooth_y"] = smooth_field(datadict["velocity_y"], sigma)
        datadict["velocity_smooth_z"] = smooth_field(datadict["velocity_z"], sigma)

        # Compute the turbulent pressure
        datadict["density_smooth"] = smooth_field(datadict["density"], sigma)
        pressure_turb = compute_turbulent_pressure(
            np.array(
                [datadict["velocity_x"], datadict["velocity_y"], datadict["velocity_z"]]
            ),
            np.array(
                [
                    datadict["velocity_smooth_x"],
                    datadict["velocity_smooth_y"],
                    datadict["velocity_smooth_z"],
                ]
            ),
            datadict["density_smooth"],
        )

        ### Compute the force from turbulent pressure
        force_turb = compute_force_from_pressure(pressure_turb, datadict["density"])

        # Compute the smoothed radial velocity
        datadict["vr_smooth"] = smooth_field(datadict["vr"], sigma)

        # Compute the radial gradient of the smoothed radial velocity
        vr_smooth_grad = np.gradient(datadict["vr_smooth"], sigma)
        vr_smooth_grad_mag = np.sqrt(
            vr_smooth_grad[0] ** 2 + vr_smooth_grad[1] ** 2 + vr_smooth_grad[2] ** 2
        )

        # Compute the ram pressure
        ram_pressure = compute_ram_pressure(
            vr_smooth_grad_mag, datadict["density_smooth"], grid_resolution
        )

        ### Compute the force from ram pressure
        force_ram = compute_force_from_pressure(ram_pressure, datadict["density"])

        # Compute the smoothed vtheta and vphi
        datadict["vtheta_smooth"] = smooth_field(datadict["vtheta"], sigma)
        datadict["vphi_smooth"] = smooth_field(datadict["vphi"], sigma)

        ### Compute the rotational force
        force_rot = compute_rotational_force(
            datadict["vtheta_smooth"], datadict["vphi_smooth"], grid_resolution
        )

        ### Compute gravitational force
        enclosed_mass = np.interp(
            datadict["radius"] / 1.0e3, ap.radiusList, ap.massList
        )

        force_grav = (
            phc.G * (enclosed_mass * u.Msun) / (datadict["radius"] * u.kpc) ** 2
        )
