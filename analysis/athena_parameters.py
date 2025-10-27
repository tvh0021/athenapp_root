# Relevant global variables + functions for Athena++ data analysis

import numpy as np
import pandas as pd
import yt
import yt.units as u
from yt import derived_field
from yt import physical_constants as phc
import os

cwd = os.path.dirname(os.path.abspath(__file__))
# print(cwd)

# yt.enable_parallelism()

lookup_units = {
    "x": "kpc",
    "y": "kpc",
    "z": "kpc",
    "number_density": "cm**-3",
    "pressure": "Pa",
    "density": "g/cm**3",
    "temperature": "K",
    "normalized_angular_momentum_x": "",
    "normalized_angular_momentum_y": "",
    "normalized_angular_momentum_z": "",
    "radius": "kpc",
    "vr": "km/s",
    "vtheta": "km/s",
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
    "pressure_gradient_x": "Pa/cm",
    "pressure_gradient_y": "Pa/cm",
    "pressure_gradient_z": "Pa/cm",
    "gravitational_potential": "cm**2/s**2",
    "bondi_ratio": "",
}


def regrid_yt(
    ds,
    fields=["velocity_x", "velocity_y", "velocity_z", "temperature"],
    dim=[256, 256, 256],
    bounding_length=[1.0, 1.0, 1.0],
    center=[0.0, 0.0, 0.0],
):
    """Regrid an Athena++ hdf5 output into a uniform grid using yt built-in function

    Args:
        ds (yt object): the dataset object from yt.load()
        fields (list[str], optional): the fields being translated to un i. Defaults to ["velocity_x", "velocity_y", "velocity_z", "temperature"].
        dim (list[int], optional): the number of cells in each direction (x, y, z). Defaults to [256, 256, 256].
        bounding_length (list[float], optional): the width of the regridded window in pc in each direction (x, y, z). Defaults to [1., 1., 1.].
        center (list[float], optional): the center of the new grid in pc. Defaults to [0., 0., 0.].

    """

    time = int(ds.current_time.to("kyr").value)
    print(f"Simulation time: {time} kyr", flush=True)

    # if "gravitational_potential_gradient" in fields:
    #     ds.add_gradient_fields(("gas", "gravitational_potential"))

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
        out_data[field] = gridded_data[field].in_units(units_list[i])

    return out_data, time


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

units_override = {
    "length_unit": (code_length, "Mpc"),
    "time_unit": (code_time, "Myr"),
    "mass_unit": (code_mass, "Msun"),
    "temperature_unit": (code_temperature, "K"),
}

cooling_curve = pd.read_csv(cwd + "/cooling_curve_solarZ.csv")
tempList = cooling_curve["Temperature (K)"].values
lambdaListCGS = cooling_curve["Lambda (erg s^-1 cm^3)"].values

SMBHMass = 6.5e9 * u.Msun
r_g = phc.G * SMBHMass / phc.c**2
# print(f"Schwarzschild radius : {r_g.to('pc')}")


# Gravity
enclosed_mass = pd.read_csv(cwd + "/enclosed_mass_M87.csv")
radiusList = enclosed_mass["R (pc)"].values
massList = enclosed_mass["Total_mass (Msun)"].values


# derived fields
@derived_field(
    name="cooling_time", sampling_type="cell", units="Myr", force_override=True
)
def _cooling_time(field, data):
    return (
        (5 / 2)
        * phc.kboltz
        * (data["gas", "temperature"].to("K"))
        / (
            (data["gas", "number_density"].to("cm**-3"))
            * np.interp(
                data["gas", "temperature"].to("K").value, tempList, lambdaListCGS
            )
            * (u.erg / u.s * u.cm**3)
        )
    )


@derived_field(
    name="sound_crossing_time", sampling_type="cell", units="Myr", force_override=True
)
def _sound_crossing_time(field, data):
    return data["gas", "dx"] / np.sqrt(
        data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    )


@derived_field(
    name="cooling_rate", sampling_type="cell", units="erg/s/cm**3", force_override=True
)
def _cooling_rate(field, data):
    return (
        data["gas", "number_density"].to("cm**-3") ** 2
        * np.interp(data["gas", "temperature"].to("K").value, tempList, lambdaListCGS)
        * (u.erg / u.s * u.cm**3)
    )


@derived_field(name="cooling_ratio", sampling_type="cell", force_override=True)
def _cooling_ratio(field, data):
    return data["gas", "cooling_time"] / data["gas", "sound_crossing_time"]


@derived_field(
    name="keplerian_speed", sampling_type="cell", units="Mpc/Myr", force_override=True
)
def _keplerian_speed(field, data):
    return np.sqrt(phc.G * SMBHMass / data["index", "radius"])


@derived_field(
    name="keplerian_specific_angular_momentum",
    sampling_type="cell",
    units="Mpc**2/Myr",
    force_override=True,
)
def _keplerian_specific_angular_momentum(field, data):
    return data["gas", "keplerian_speed"] * data["index", "radius"]


@derived_field(
    name="normalized_specific_angular_momentum_x",
    sampling_type="cell",
    force_override=True,
)
def _normalized_specific_angular_momentum_x(field, data):
    return (
        data["gas", "specific_angular_momentum_x"]
        / data["gas", "keplerian_specific_angular_momentum"]
    )


@derived_field(
    name="normalized_specific_angular_momentum_y",
    sampling_type="cell",
    force_override=True,
)
def _normalized_specific_angular_momentum_y(field, data):
    return (
        data["gas", "specific_angular_momentum_y"]
        / data["gas", "keplerian_specific_angular_momentum"]
    )


@derived_field(
    name="normalized_specific_angular_momentum_z",
    sampling_type="cell",
    force_override=True,
)
def _normalized_specific_angular_momentum_z(field, data):
    return (
        data["gas", "specific_angular_momentum_z"]
        / data["gas", "keplerian_specific_angular_momentum"]
    )


@derived_field(
    name="keplerian_angular_momentum", sampling_type="cell", force_override=True
)
def _keplerian_angular_momentum(field, data):
    return data["gas", "mass"] * data["gas", "keplerian_specific_angular_momentum"]


@derived_field(
    name="normalized_angular_momentum_x", sampling_type="cell", force_override=True
)
def _normalized_angular_momentum_x(field, data):
    return data["gas", "angular_momentum_x"] / data["gas", "keplerian_angular_momentum"]


@derived_field(
    name="normalized_angular_momentum_y", sampling_type="cell", force_override=True
)
def _normalized_angular_momentum_y(field, data):
    return data["gas", "angular_momentum_y"] / data["gas", "keplerian_angular_momentum"]


@derived_field(
    name="normalized_angular_momentum_z", sampling_type="cell", force_override=True
)
def _normalized_angular_momentum_z(field, data):
    return data["gas", "angular_momentum_z"] / data["gas", "keplerian_angular_momentum"]


@derived_field(
    name="free_fall_time", sampling_type="cell", units="Myr", force_override=True
)
def _free_fall_time(field, data):
    return (
        np.pi
        / 2
        * data["index", "radius"] ** (3 / 2)
        / np.sqrt(2 * phc.G * (SMBHMass + data["gas", "mass"]))
    )


@derived_field(name="free_fall_ratio", sampling_type="cell", force_override=True)
def _free_fall_ratio(field, data):
    return data["gas", "cooling_time"] / data["gas", "free_fall_time"]


@derived_field(
    name="density_squared",
    sampling_type="cell",
    units="msun**2*Mpc**-6",
    force_override=True,
)
def _density_squared(field, data):
    return data["gas", "density"] ** 2


@derived_field(name="angle_theta", sampling_type="cell", force_override=True)
def _angle_theta(field, data):
    return np.arccos(data["index", "z"] / data["index", "radius"])


@derived_field(name="angle_phi", sampling_type="cell", force_override=True)
def _angle_phi(field, data):
    return np.arctan2(data["index", "y"], data["index", "x"])


@derived_field(name="vr", sampling_type="cell", units="km/s", force_override=True)
def _vr(field, data):
    return (
        data["gas", "velocity_x"] * data["index", "x"]
        + data["gas", "velocity_y"] * data["index", "y"]
        + data["gas", "velocity_z"] * data["index", "z"]
    ) / data["index", "radius"]


@derived_field(name="vtheta", sampling_type="cell", units="km/s", force_override=True)
def _vtheta(field, data):
    return (
        data["gas", "velocity_x"]
        * np.cos(data["gas", "angle_theta"])
        * np.cos(data["gas", "angle_phi"])
        + data["gas", "velocity_y"]
        * np.cos(data["gas", "angle_theta"])
        * np.sin(data["gas", "angle_phi"])
        - data["gas", "velocity_z"] * np.sin(data["gas", "angle_theta"])
    )


@derived_field(name="vphi", sampling_type="cell", units="km/s", force_override=True)
def _vphi(field, data):
    return data["gas", "velocity_y"] * np.cos(data["gas", "angle_phi"]) - data[
        "gas", "velocity_x"
    ] * np.sin(data["gas", "angle_phi"])


@derived_field(name="vtangent", sampling_type="cell", units="km/s", force_override=True)
def _vtangent(field, data):
    return np.sqrt(data["gas", "velocity_magnitude"] ** 2 - data["gas", "vr"] ** 2)


@derived_field(
    name="gravitational_potential",
    sampling_type="cell",
    units="cm**2/s**2",
    force_override=True,
)
def _gravitational_potential(field, data):
    return phc.G * SMBHMass / data["index", "radius"]


@derived_field(
    name="bondi_ratio", sampling_type="cell", units="dimensionless", force_override=True
)
def _bondi_ratio(field, data):
    return (
        data["gas", "gravitational_potential"] / data["gas", "specific_thermal_energy"]
    )


@derived_field(
    name="H_nuclei_density", sampling_type="cell", units="cm**-3", force_override=True
)
def _H_nuclei_density(field, data):
    X_H = 0.7  # Hydrogen mass fraction
    m_H = 1.6726e-24 * u.g  # g
    return data["gas", "density"].to("g/cm**3") * X_H / m_H


@derived_field(
    name="El_number_density", sampling_type="cell", units="cm**-3", force_override=True
)
def _El_number_density(field, data):
    return data["gas", "H_nuclei_density"] * 1.2
