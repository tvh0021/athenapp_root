## By Trung Ha, 2024
## This file is used to generate videos of the simulation data
## Example command: python ~/git_repos/athenapp_root/analysis/get_plots.py --base_ext "M87.out2." --snapshots 50 500
##  --window 0.5 5 50 --dimension 'x' 'z' --types 'temperature' 'number_density' --method 'slice'


import yt
import numpy as np
from yt import physical_constants as phc
import yt.units as units

import sys
import os

# sys.path.append("/home1/08520/trungha/git_repos/ehtplot")
sys.path.append("/Users/tvh0021//git_repos/ehtplot")
import ehtplot.color

import argparse
from multiprocessing import Pool
from multiprocessing import cpu_count

import athena_parameters as ap

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

# unit override
units_override = {
    "length_unit": (code_length, "Mpc"),
    "time_unit": (code_time, "Myr"),
    "mass_unit": (code_mass, "Msun"),
    "temperature_unit": (code_temperature, "K"),
}


# additional functions
kpcCGS = 3.08567758096e21
MpcCGS = kpcCGS * 1.0e3
yearCGS = 31557600.0
MyrCGS = yearCGS * 1.0e6
solarMassCGS = 1.988e33
boltzmannConstCGS = 1.3806488e-16

boltzmannConstAstronomical = boltzmannConstCGS / (solarMassCGS * MpcCGS**2 * MyrCGS**-2)
codeBoltzmannConst = boltzmannConstAstronomical / (code_energy / code_temperature)

logTemperatureArray = np.array(
    [
        3.8,
        3.84,
        3.88,
        3.92,
        3.96,
        4.0,
        4.04,
        4.08,
        4.12,
        4.16,
        4.2,
        4.24,
        4.28,
        4.32,
        4.36,
        4.4,
        4.44,
        4.48,
        4.52,
        4.56,
        4.6,
        4.64,
        4.68,
        4.72,
        4.76,
        4.8,
        4.84,
        4.88,
        4.92,
        4.96,
        5.0,
        5.04,
        5.08,
        5.12,
        5.16,
        5.2,
        5.24,
        5.28,
        5.32,
        5.36,
        5.4,
        5.44,
        5.48,
        5.52,
        5.56,
        5.6,
        5.64,
        5.68,
        5.72,
        5.76,
        5.8,
        5.84,
        5.88,
        5.92,
        5.96,
        6.0,
        6.04,
        6.08,
        6.12,
        6.16,
        6.2,
        6.24,
        6.28,
        6.32,
        6.36,
        6.4,
        6.44,
        6.48,
        6.52,
        6.56,
        6.6,
        6.64,
        6.68,
        6.72,
        6.76,
        6.8,
        6.84,
        6.88,
        6.92,
        6.96,
        7.0,
        7.04,
        7.08,
        7.12,
        7.16,
        7.2,
        7.24,
        7.28,
        7.32,
        7.36,
        7.4,
        7.44,
        7.48,
        7.52,
        7.56,
        7.6,
        7.64,
        7.68,
        7.72,
        7.76,
        7.8,
        7.84,
        7.88,
        7.92,
        7.96,
        8.0,
        8.04,
        8.08,
        8.12,
        8.16,
    ]
)
logEmissivityHydroArray = np.array(
    [
        -30.6104,
        -29.4107,
        -28.4601,
        -27.5743,
        -26.3766,
        -25.289,
        -24.2684,
        -23.3834,
        -22.5977,
        -21.9689,
        -21.5972,
        -21.4615,
        -21.4789,
        -21.5497,
        -21.6211,
        -21.6595,
        -21.6426,
        -21.5688,
        -21.4771,
        -21.3755,
        -21.2693,
        -21.1644,
        -21.0658,
        -20.9778,
        -20.8986,
        -20.8281,
        -20.77,
        -20.7223,
        -20.6888,
        -20.6739,
        -20.6815,
        -20.7051,
        -20.7229,
        -20.7208,
        -20.7058,
        -20.6896,
        -20.6797,
        -20.6749,
        -20.6709,
        -20.6748,
        -20.7089,
        -20.8031,
        -20.9647,
        -21.1482,
        -21.2932,
        -21.3767,
        -21.4129,
        -21.4291,
        -21.4538,
        -21.5055,
        -21.574,
        -21.63,
        -21.6615,
        -21.6766,
        -21.6886,
        -21.7073,
        -21.7304,
        -21.7491,
        -21.7607,
        -21.7701,
        -21.7877,
        -21.8243,
        -21.8875,
        -21.9738,
        -22.0671,
        -22.1537,
        -22.2265,
        -22.2821,
        -22.3213,
        -22.3462,
        -22.3587,
        -22.3622,
        -22.359,
        -22.3512,
        -22.342,
        -22.3342,
        -22.3312,
        -22.3346,
        -22.3445,
        -22.3595,
        -22.378,
        -22.4007,
        -22.4289,
        -22.4625,
        -22.4995,
        -22.5353,
        -22.5659,
        -22.5895,
        -22.6059,
        -22.6161,
        -22.6208,
        -22.6213,
        -22.6184,
        -22.6126,
        -22.6045,
        -22.5945,
        -22.5831,
        -22.5707,
        -22.5573,
        -22.5434,
        -22.5287,
        -22.514,
        -22.4992,
        -22.4844,
        -22.4695,
        -22.4543,
        -22.4392,
        -22.4237,
        -22.4087,
        -22.3928,
    ]
)

tempList = np.logspace(3.0, 9.0, 200)


def emissivityFromTemperature(temperature):
    # Real emissivityCGS, emissivityAstronomical;
    logTemperature = np.log10(temperature)
    if logTemperature <= 4.2:  # Koyama & Inutsuka (2002)
        emissivityCGS = 2.0e-19 * np.exp(
            -1.184e5 / (temperature + 1.0e3)
        ) + 2.8e-28 * np.sqrt(temperature) * np.exp(-92.0 / temperature)
    elif logTemperature > 8.15:  # Schneider & Robertson (2018)
        emissivityCGS = 10.0 ** (0.45 * logTemperature - 26.065)
    else:  # Schure+09
        emissivityCGS = 10.0 ** (
            np.interp(logTemperature, logTemperatureArray, logEmissivityHydroArray)
        )

    emissivityAstronomical = emissivityCGS / (
        solarMassCGS * pow(MpcCGS, 5) * pow(MyrCGS, -3)
    )

    return emissivityCGS


lambdaListCGS = np.zeros(200)
for i, temp in enumerate(tempList):
    lambdaListCGS[i] = emissivityFromTemperature(temp)

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

interpLambdaFunction = interp1d(tempList, lambdaListCGS)

# from yt import derived_field

# SMBHMass = 6.5e9


# @derived_field(
#     name="cooling_time", sampling_type="cell", units="Myr", force_override=True
# )
# def _cooling_time(field, data):
#     return (
#         (5 / 2)
#         * phc.kboltz
#         * (data["gas", "temperature"].to("K"))
#         / (
#             (data["gas", "number_density"].to("cm**-3"))
#             * np.interp(
#                 data["gas", "temperature"].to("K").value, tempList, lambdaListCGS
#             )
#             * (units.erg / units.s * units.cm**3)
#         )
#     )


# @derived_field(
#     name="sound_crossing_time", sampling_type="cell", units="Myr", force_override=True
# )
# def _sound_crossing_time(field, data):
#     return data["gas", "dx"] / np.sqrt(
#         data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
#     )


# @derived_field(
#     name="cooling_rate", sampling_type="cell", units="erg/s/cm**3", force_override=True
# )
# def _cooling_rate(field, data):
#     return (
#         data["gas", "number_density"].to("cm**-3") ** 2
#         * np.interp(data["gas", "temperature"].to("K").value, tempList, lambdaListCGS)
#         * (units.erg / units.s * units.cm**3)
#     )


# @derived_field(name="cooling_ratio", sampling_type="cell", force_override=True)
# def _cooling_ratio(field, data):
#     return data["gas", "cooling_time"] / data["gas", "sound_crossing_time"]


# @derived_field(
#     name="keplerian_speed", sampling_type="cell", units="Mpc/Myr", force_override=True
# )
# def _keplerian_speed(field, data):
#     return np.sqrt(phc.G * SMBHMass * phc.msun / data["index", "radius"])


# @derived_field(
#     name="keplerian_specific_angular_momentum",
#     sampling_type="cell",
#     units="Mpc**2/Myr",
#     force_override=True,
# )
# def _keplerian_specific_angular_momentum(field, data):
#     return data["gas", "keplerian_speed"] * data["index", "radius"]


# @derived_field(
#     name="normalized_specific_angular_momentum_x",
#     sampling_type="cell",
#     force_override=True,
# )
# def _normalized_specific_angular_momentum_x(field, data):
#     return (
#         data["gas", "specific_angular_momentum_x"]
#         / data["gas", "keplerian_specific_angular_momentum"]
#     )


# @derived_field(
#     name="normalized_specific_angular_momentum_y",
#     sampling_type="cell",
#     force_override=True,
# )
# def _normalized_specific_angular_momentum_y(field, data):
#     return (
#         data["gas", "specific_angular_momentum_y"]
#         / data["gas", "keplerian_specific_angular_momentum"]
#     )


# @derived_field(
#     name="normalized_specific_angular_momentum_z",
#     sampling_type="cell",
#     force_override=True,
# )
# def _normalized_specific_angular_momentum_z(field, data):
#     return (
#         data["gas", "specific_angular_momentum_z"]
#         / data["gas", "keplerian_specific_angular_momentum"]
#     )


# @derived_field(
#     name="keplerian_angular_momentum", sampling_type="cell", force_override=True
# )
# def _keplerian_angular_momentum(field, data):
#     return data["gas", "mass"] * data["gas", "keplerian_specific_angular_momentum"]


# @derived_field(
#     name="normalized_angular_momentum_x", sampling_type="cell", force_override=True
# )
# def _normalized_angular_momentum_x(field, data):
#     return data["gas", "angular_momentum_x"] / data["gas", "keplerian_angular_momentum"]


# @derived_field(
#     name="normalized_angular_momentum_y", sampling_type="cell", force_override=True
# )
# def _normalized_angular_momentum_y(field, data):
#     return data["gas", "angular_momentum_y"] / data["gas", "keplerian_angular_momentum"]


# @derived_field(
#     name="normalized_angular_momentum_z", sampling_type="cell", force_override=True
# )
# def _normalized_angular_momentum_z(field, data):
#     return data["gas", "angular_momentum_z"] / data["gas", "keplerian_angular_momentum"]


# @derived_field(
#     name="free_fall_time", sampling_type="cell", units="Myr", force_override=True
# )
# def _free_fall_time(field, data):
#     return (
#         np.pi
#         / 2
#         * data["index", "radius"] ** (3 / 2)
#         / np.sqrt(2 * phc.G * (SMBHMass * phc.msun + data["gas", "mass"]))
#     )


# @derived_field(name="free_fall_ratio", sampling_type="cell", force_override=True)
# def _free_fall_ratio(field, data):
#     return data["gas", "cooling_time"] / data["gas", "free_fall_time"]


# @derived_field(
#     name="density_squared",
#     sampling_type="cell",
#     units="msun**2*Mpc**-6",
#     force_override=True,
# )
# def _density_squared(field, data):
#     return data["gas", "density"] ** 2


# @derived_field(name="angle_theta", sampling_type="cell", force_override=True)
# def _angle_theta(field, data):
#     return np.arccos(data["index", "z"] / data["index", "radius"])


# @derived_field(name="angle_phi", sampling_type="cell", force_override=True)
# def _angle_phi(field, data):
#     return np.arctan2(data["index", "y"], data["index", "x"])


# @derived_field(name="vr", sampling_type="cell", units="km/s", force_override=True)
# def _vr(field, data):
#     return (
#         data["gas", "velocity_x"] * data["index", "x"]
#         + data["gas", "velocity_y"] * data["index", "y"]
#         + data["gas", "velocity_z"] * data["index", "z"]
#     ) / data["index", "radius"]


# @derived_field(name="vtheta", sampling_type="cell", units="", force_override=True)
# def _vtheta(field, data):
#     return (
#         data["gas", "vr"] * data["gas", "angle_theta"] - data["gas", "velocity_z"]
#     ) / (data["index", "radius"] * np.sin(data["gas", "angle_theta"]))


# @derived_field(name="vphi", sampling_type="cell", units="km/s", force_override=True)
# def _vphi(field, data):
#     return data["gas", "velocity_y"] * np.cos(data["gas", "angle_phi"]) - data[
#         "gas", "velocity_x"
#     ] * np.sin(data["gas", "angle_phi"])


# @derived_field(name="vtangent", sampling_type="cell", units="km/s", force_override=True)
# def _vtangent(field, data):
#     return np.sqrt(data["gas", "velocity_magnitude"] ** 2 - data["gas", "vr"] ** 2)


# @derived_field(
#     name="gravitational_potential",
#     sampling_type="cell",
#     units="cm**2/s**2",
#     force_override=True,
# )
# def _gravitational_potential(field, data):
#     return phc.G * (SMBHMass * phc.msun) / data["index", "radius"]


# @derived_field(
#     name="bondi_ratio", sampling_type="cell", units="dimensionless", force_override=True
# )
# def _bondi_ratio(field, data):
#     return (
#         data["gas", "gravitational_potential"] / data["gas", "specific_thermal_energy"]
#     )


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


# lookup_units = {
#     "number_density": "cm**-3",
#     "pressure": "Pa",
#     "density": "g/cm**3",
#     "temperature": "K",
#     "normalized_angular_momentum_z": "",
#     "vr": "km/s",
#     "bondi_ratio": "",
# }
lookup_range = {
    "number_density": (1.0e-2, 1.0e7),
    "pressure": (1.0e-13, 1.0e-11),
    "density": (1.0e-26, 1.0e-23),
    "temperature": (2.0e5, 1.0e9),
    "normalized_angular_momentum_z": (-1.2, 1.2),
    "vr": (-1.0e3, 1.0e3),
    "bondi_ratio": (1.0e-2, 9.0e1),
}
lookup_cmap = {
    "number_density": "afmhot_10us",
    "pressure": "viridis",
    "density": "magma",
    "temperature": "RdBu_r",
    "normalized_angular_momentum_z": "jet",
    "vr": "viridis",
    "bondi_ratio": "Pastels",
}


def make_plots(
    location: str,
    base_ext: str,
    dimensions: list[str],
    types: list[str],
    window: list[tuple],
    start_nfile: int,
    stop_nfile: int,
    method: str,
    velocity_on: bool = False,
):

    for i in range(start_nfile, stop_nfile + 1):
        ind = "%05d" % (i)

        ds = yt.load(makeFilename(location, base_ext, i), units_override=units_override)

        for dim in dimensions:
            if (dim == "x") & (velocity_on):
                restrict_dim = ["y", "z"]
            elif (dim == "y") & (velocity_on):
                restrict_dim = ["z", "x"]
            elif (dim == "z") & (velocity_on):
                restrict_dim = ["x", "y"]
            elif velocity_on & (method == "projection"):
                return "Cannot use custom projection with velocity"
            for plot_type in types:
                for zoom in window:
                    if method == "projection":
                        # custom projection
                        L = [0.5, 0.35, 0.3]  # vector normal to cutting plane
                        if dim == "c":
                            p = yt.ProjectionPlot(
                                ds,
                                L,
                                ("gas", plot_type),
                                width=zoom,
                                buff_size=(1500, 1500),
                                weight_field=("gas", "cooling_rate"),
                            )
                        else:
                            p = yt.ProjectionPlot(
                                ds,
                                dim,
                                ("gas", plot_type),
                                width=zoom,
                                buff_size=(1500, 1500),
                                weight_field=("gas", "cooling_rate"),
                            )
                        p.set_unit(("gas", plot_type), ap.lookup_units[plot_type])
                        p.set_zlim(
                            plot_type,
                            zmin=lookup_range[plot_type][0],
                            zmax=lookup_range[plot_type][1],
                        )  # 1e0 to 5e6 if meso scale
                        p.set_cmap(plot_type, lookup_cmap[plot_type])
                        p.annotate_scale(corner="lower_left")
                        p.annotate_timestamp(
                            time_unit="Myr",
                            corner="upper_right",
                            time_format="t = {time:.3f} {units}",
                        )
                        p.annotate_title(f"{dim} {plot_type} projection")
                        if velocity_on:
                            p.annotate_streamlines(
                                ("gas", f"velocity_{restrict_dim[0]}"),
                                ("gas", f"velocity_{restrict_dim[1]}"),
                                factor=2,
                                density=3,
                            )
                        p.hide_axes()
                        if dim == "y":
                            p.swap_axes()
                        p.save(
                            location
                            + f"proj,{dim},{plot_type},{zoom[0]}/{base_ext}{ind}.png"
                        )

                    elif method == "slice":
                        s = yt.SlicePlot(
                            ds,
                            dim,
                            ("gas", plot_type),
                            width=zoom,
                            buff_size=(1500, 1500),
                        )
                        s.set_unit(("gas", plot_type), ap.lookup_units[plot_type])
                        if plot_type == "normalized_angular_momentum_z":
                            s.set_log(plot_type, linthresh=0.75)
                        if plot_type == "vr":
                            s.set_log(plot_type, linthresh=1.0e2)
                        s.set_zlim(
                            plot_type,
                            zmin=lookup_range[plot_type][0],
                            zmax=lookup_range[plot_type][1],
                        )
                        s.set_cmap(plot_type, lookup_cmap[plot_type])
                        s.annotate_scale(corner="lower_left")
                        s.annotate_timestamp(
                            time_unit="Myr",
                            corner="upper_right",
                            time_format="t = {time:.3f} {units}",
                        )
                        s.annotate_title(f"{dim} {plot_type} slice")
                        s.hide_axes()
                        if dim == "y":
                            s.swap_axes()
                        if velocity_on:
                            s.annotate_streamlines(
                                ("gas", f"velocity_{restrict_dim[0]}"),
                                ("gas", f"velocity_{restrict_dim[1]}"),
                                factor=2,
                                density=3,
                            )
                        s.save(
                            location
                            + f"slice,{dim},{plot_type},{zoom[0]}/{base_ext}{ind}.png"
                        )


def get_multiple_snapshots(
    location: str,
    base_ext: str,
    dimensions: list[str],
    types: list[str],
    window: list[tuple],
    start_nfile: int,
    stop_nfile: int,
    method: str,
    nproc: int,
    velocity_on: bool = False,
):
    """With multiprocessing, generate multiple snapshots of the simulation data at once and save them to the specified location

    Args:
        location (str): path to the simulation data
        base_ext (str): base extension of the snapshot files
        dimensions (list[str]): a list of dimensions to plot ('x', 'y', 'z', 'c' for custom projection)
        types (list[str]): a list of physical quantities to plot ('temperature', 'number_density', 'pressure', 'density', 'normalized_angular_momentum_z', 'vr')
        window (list[tuple]): a list of window sizes for the plots in kpc
        start_nfile (int): starting snapshot number
        stop_nfile (int): ending snapshot number
        method (str): slice or projection
        nproc (int): number of cores to use for multiprocessing
        velocity_on (bool, optional): whether velocity streamlines are plotted. Defaults to False.

    Returns:
        None: all snapshots are saved to the specified location
    """

    # kernel_size = kernelA.shape[0]
    # extended_feature = np.pad(feature_array, ((kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2)), mode='wrap')
    print("Saving snapshots using {} cores".format(nproc), flush=True)

    with Pool(processes=nproc) as p:
        items = [
            (location, base_ext, dimensions, types, window, k, k, method, velocity_on)
            for k in range(start_nfile, stop_nfile + 1)
        ]

        for k in enumerate(p.starmap(make_plots, items)):
            print(f"Snapshot {k[0]} done", flush=True)

    print("All snapshots saved to " + location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation data video generator")
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
    parser.add_argument(
        "--window",
        type=float,
        nargs="+",
        action="store",
        help="window size for the plots in kpc",
    )
    parser.add_argument(
        "--dimensions", type=str, nargs="+", action="store", help="dimensions to plot"
    )
    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        action="store",
        help="physical quantities to plot",
    )
    parser.add_argument("--method", type=str, help="slice or projection")
    parser.add_argument(
        "--nproc",
        type=int,
        default=cpu_count(),
        help="number of cores to use for multiprocessing",
        required=False,
    )
    parser.add_argument(
        "--velocity_on",
        action="store_true",
        help="velocity streamlines on or off",
        required=False,
    )

    args = parser.parse_args()

    path = args.path + "/"
    base_ext = args.base_ext
    (start_nfile, stop_nfile) = args.snapshots
    window = list(args.window)
    for i in range(len(window)):
        window[i] = (window[i], "kpc")
    dimensions = args.dimensions
    types = args.types
    method = args.method
    velocity_on = args.velocity_on
    nproc = args.nproc

    print("Parameters: ")
    print(f"Path: {path}")
    print(f"Base extension: {base_ext}")
    print(f"Snapshots: {start_nfile} to {stop_nfile}")
    print(f"Window: {window}")
    print(f"Dimensions: {dimensions}")
    print(f"Types: {types}")
    print(f"Method: {method}")
    print(f"Velocity on: {velocity_on}")

    # create directories for the plots
    for dim in dimensions:
        for plot_type in types:
            for zoom in window:
                contents = os.listdir(path)
                if method == "projection":
                    if (f"proj,{dim},{plot_type},{zoom[0]}") not in contents:
                        os.mkdir(path + f"proj,{dim},{plot_type},{zoom[0]}")
                elif method == "slice":
                    if (f"slice,{dim},{plot_type},{zoom[0]}") not in contents:
                        os.mkdir(path + f"slice,{dim},{plot_type},{zoom[0]}")

    # make_plots(path, base_ext, dimensions, types, window, start_nfile, stop_nfile, method, velocity_on)
    get_multiple_snapshots(
        path,
        base_ext,
        dimensions,
        types,
        window,
        start_nfile,
        stop_nfile,
        method,
        nproc,
        velocity_on,
    )
