# Relevant global variables + functions for Athena++ data analysis

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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


def K_to_keV(temperature):
    return temperature * boltzmannConstCGS / 1.60218e-9


def keV_to_K(temperature_keV):
    return temperature_keV * 1.60218e-9 / boltzmannConstCGS


def closest_value(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


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

tempList = np.logspace(3.0, 12.0, 300)


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

    # emissivityAstronomical = emissivityCGS / (solarMassCGS * pow(MpcCGS, 5) * pow(MyrCGS, -3))

    return emissivityCGS


lambdaListCGS = np.zeros(300)
for i, temp in enumerate(tempList):
    lambdaListCGS[i] = emissivityFromTemperature(temp)

interpLambdaFunction = interp1d(tempList, lambdaListCGS)

SMBHMass = 6.5e9 * u.Msun
r_g = phc.G * SMBHMass / phc.c**2


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
    name="PdV", sampling_type="cell", units="erg/s/cm**3", force_override=True
)
def _PdV(field, data):
    return -1.0 * data["gas", "pressure"] * data["gas", "velocity_divergence"]


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
    name="normalized_specific_angular_momentum_magnitude",
    sampling_type="cell",
    force_override=True,
)
def _normalized_specific_angular_momentum_magnitude(field, data):
    return (
        np.sqrt(
            data["gas", "specific_angular_momentum_x"] ** 2
            + data["gas", "specific_angular_momentum_y"] ** 2
            + data["gas", "specific_angular_momentum_z"] ** 2
        )
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
    # return (data["gas", "vr"] * data["gas", "angle_theta"] - data["gas", "velocity_z"]) / (data["index", "radius"] * np.sin(data["gas", "angle_theta"]))
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
    name="rotational_velocity", sampling_type="cell", units="km/s", force_override=True
)
def _rotational_velocity(field, data):
    r_vec = np.array(
        [
            data["gas", "x"].to("Mpc"),
            data["gas", "y"].to("Mpc"),
            data["gas", "z"].to("Mpc"),
        ]
    ).T
    L_hat = np.array(
        [
            data["gas", "normalized_specific_angular_momentum_x"],
            data["gas", "normalized_specific_angular_momentum_y"],
            data["gas", "normalized_specific_angular_momentum_z"],
        ]
    ).T

    # compute r_cyl, which is the cylindrical radius in the plane perpendicular to L_hat
    r_parallel = np.sum(r_vec * L_hat, axis=-1)
    r_cyl = np.sqrt(np.sum(r_vec**2, axis=-1) - r_parallel**2)
    r_cyl *= u.Mpc
    return data["gas", "specific_angular_momentum_magnitude"] / r_cyl


@derived_field(name="accretion_parameter", sampling_type="cell", force_override=True)
def _accretion_parameter(field, data):
    return data["gas", "vr"] / data["gas", "vtangent"]


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


@derived_field(
    name="local_bondi_radius", sampling_type="cell", units="pc", force_override=True
)
def _local_bondi_radius(field, data):
    return 2 * phc.G * SMBHMass / (data["sound_speed"] ** 2)


@derived_field(
    name="in_bondi_region", sampling_type="cell", units="", force_override=True
)
def _in_bondi_region(field, data):
    return data["local_bondi_radius"] / data["radius"]


@derived_field(
    name="bondi_accretion_rate",
    sampling_type="cell",
    units="Msun/yr",
    force_override=True,
)
def _bondi_accretion_rate(field, data):
    rho = data["density"]
    cs = data["sound_speed"]
    bondi_rate = (4 * np.pi * phc.G**2 * SMBHMass**2 * rho) / (cs**3)
    return bondi_rate.to("Msun/yr")


@derived_field(
    name="circularization_radius", sampling_type="cell", units="pc", force_override=True
)
def _circularization_radius(field, data):
    return (data["radius"] * data["vtangent"]) ** 2 / (phc.G * SMBHMass)


@derived_field(
    name="in_circularization_region",
    sampling_type="cell",
    units="",
    force_override=True,
)
def _in_circularization_region(field, data):
    return data["circularization_radius"] / data["radius"]


@derived_field(name="r_z", sampling_type="cell", units="pc", force_override=True)
def _r_z(field, data):
    return np.sqrt(data["index", "x"] ** 2 + data["index", "y"] ** 2)


@derived_field(name="r_x", sampling_type="cell", units="pc", force_override=True)
def _r_x(field, data):
    return np.sqrt(data["index", "y"] ** 2 + data["index", "z"] ** 2)


@derived_field(name="r_y", sampling_type="cell", units="pc", force_override=True)
def _r_y(field, data):
    return np.sqrt(data["index", "z"] ** 2 + data["index", "x"] ** 2)
