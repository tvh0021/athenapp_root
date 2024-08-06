## By Trung Ha, 2024
## This script is used to extract data from the output of the simulation

import yt
import numpy as np
from yt import physical_constants as phc
import yt.units as units

import pickle
import os

import argparse
from multiprocessing import Pool
from multiprocessing import cpu_count

#unit conversions
code_length = 0.064
code_mass = 1.
code_temperature = 1.
code_time = 10.
code_area = code_length**2
code_volume = code_length**3
code_velocity = code_length / code_time
code_momentum = code_velocity * code_mass
code_density = code_mass / code_volume
code_acceleration = code_velocity / code_time
code_force = code_mass * code_acceleration
code_pressure = code_force / code_area
code_energy = code_force * code_length

#unit override
units_override = {"length_unit": (code_length, "Mpc"),"time_unit": (code_time, "Myr"),"mass_unit": (code_mass, "Msun"),"temperature_unit": (code_temperature, "K")}


# additional functions
kpcCGS = 3.08567758096e+21
MpcCGS = kpcCGS * 1.e3
yearCGS = 31557600.0
MyrCGS = yearCGS * 1.e6
solarMassCGS = 1.988e33
boltzmannConstCGS = 1.3806488e-16

boltzmannConstAstronomical = boltzmannConstCGS / (solarMassCGS * MpcCGS**2 * MyrCGS**-2)
codeBoltzmannConst = boltzmannConstAstronomical / (code_energy / code_temperature)

logTemperatureArray = np.array([3.8, 3.84, 3.88, 3.92, 3.96, 4., 4.04, 4.08, 4.12, 4.16, 4.2,
                                          4.24, 4.28, 4.32, 4.36, 4.4, 4.44, 4.48, 4.52, 4.56, 4.6, 4.64,
                                          4.68, 4.72, 4.76, 4.8, 4.84, 4.88, 4.92, 4.96, 5., 5.04, 5.08,
                                          5.12, 5.16, 5.2, 5.24, 5.28, 5.32, 5.36, 5.4, 5.44, 5.48, 5.52,
                                          5.56, 5.6, 5.64, 5.68, 5.72, 5.76, 5.8, 5.84, 5.88, 5.92, 5.96,
                                          6., 6.04, 6.08, 6.12, 6.16, 6.2, 6.24, 6.28, 6.32, 6.36, 6.4,
                                          6.44, 6.48, 6.52, 6.56, 6.6, 6.64, 6.68, 6.72, 6.76, 6.8, 6.84,
                                          6.88, 6.92, 6.96, 7., 7.04, 7.08, 7.12, 7.16, 7.2, 7.24, 7.28,
                                          7.32, 7.36, 7.4, 7.44, 7.48, 7.52, 7.56, 7.6, 7.64, 7.68, 7.72,
                                          7.76, 7.8, 7.84, 7.88, 7.92, 7.96, 8., 8.04, 8.08, 8.12, 8.16])
logEmissivityHydroArray = np.array([-30.6104, -29.4107, -28.4601, -27.5743, -26.3766, -25.289,
                                              -24.2684, -23.3834, -22.5977, -21.9689, -21.5972, -21.4615,
                                              -21.4789, -21.5497, -21.6211, -21.6595, -21.6426, -21.5688,
                                              -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
                                              -20.8986, -20.8281, -20.77, -20.7223, -20.6888, -20.6739,
                                              -20.6815, -20.7051, -20.7229, -20.7208, -20.7058, -20.6896,
                                              -20.6797, -20.6749, -20.6709, -20.6748, -20.7089, -20.8031,
                                              -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
                                              -21.4538, -21.5055, -21.574, -21.63, -21.6615, -21.6766,
                                              -21.6886, -21.7073, -21.7304, -21.7491, -21.7607, -21.7701,
                                              -21.7877, -21.8243, -21.8875, -21.9738, -22.0671, -22.1537,
                                              -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
                                              -22.359, -22.3512, -22.342, -22.3342, -22.3312, -22.3346,
                                              -22.3445, -22.3595, -22.378, -22.4007, -22.4289, -22.4625,
                                              -22.4995, -22.5353, -22.5659, -22.5895, -22.6059, -22.6161,
                                              -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
                                              -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.514,
                                              -22.4992, -22.4844, -22.4695, -22.4543, -22.4392, -22.4237,
                                              -22.4087, -22.3928])

tempList = np.logspace(3.,9.,200)

def emissivityFromTemperature(temperature):
    #Real emissivityCGS, emissivityAstronomical;
    logTemperature = np.log10(temperature)
    if logTemperature <= 4.2: # Koyama & Inutsuka (2002)
        emissivityCGS = (2.0e-19 * np.exp(-1.184e5 / (temperature + 1.e3)) + 2.8e-28 * np.sqrt(temperature) * np.exp(-92. / temperature))
    elif logTemperature > 8.15: # Schneider & Robertson (2018)
        emissivityCGS = 10.**(0.45 * logTemperature - 26.065)
    else: # Schure+09
        emissivityCGS = 10.**(np.interp(logTemperature, logTemperatureArray, logEmissivityHydroArray))

    emissivityAstronomical = emissivityCGS / (solarMassCGS * pow(MpcCGS, 5) * pow(MyrCGS, -3))

    return emissivityCGS

lambdaListCGS = np.zeros(200)
for i, temp in enumerate(tempList):
    lambdaListCGS[i] = emissivityFromTemperature(temp)

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
interpLambdaFunction = interp1d(tempList, lambdaListCGS)

from yt import derived_field

SMBHMass = 6.5e9

@derived_field(name="cooling_time", sampling_type="cell", units="Myr", force_override=True)
def _cooling_time(field, data):
    return ( (5/2) *  phc.kboltz * (data["gas", "temperature"].in_units("K")) / ((data["gas", "number_density"].to("cm**-3")) * np.interp(data["gas", "temperature"].in_units("K").value, tempList, lambdaListCGS) * (units.erg / units.s * units.cm**3)))

@derived_field(name="sound_crossing_time", sampling_type="cell", units="Myr", force_override=True)
def _sound_crossing_time(field, data):
    return data["gas", "dx"] / np.sqrt(data.ds.gamma * data["gas", "pressure"] / data["gas", "density"])

@derived_field(name="cooling_rate", sampling_type="cell", units="erg/s/cm**3", force_override=True)
def _cooling_rate(field, data):
    return data["gas", "number_density"].in_units("cm**-3")**2 * np.interp(data["gas", "temperature"].to("K").value, tempList, lambdaListCGS) * (units.erg / units.s * units.cm**3)

@derived_field(name="cooling_ratio", sampling_type="cell", force_override=True)
def _cooling_ratio(field, data):
    return data["gas", "cooling_time"] / data["gas", "sound_crossing_time"]

@derived_field(name="keplerian_speed", sampling_type="cell", units="Mpc/Myr", force_override=True)
def _keplerian_speed(field, data):
    return np.sqrt(phc.G * SMBHMass * phc.msun / data["index", "radius"])

@derived_field(name="keplerian_specific_angular_momentum", sampling_type="cell", units="Mpc**2/Myr", force_override=True)
def _keplerian_specific_angular_momentum(field, data):
    return data["gas", "keplerian_speed"] * data["index", "radius"]

@derived_field(name="normalized_specific_angular_momentum_x", sampling_type="cell", force_override=True)
def _normalized_specific_angular_momentum_x(field, data):
    return data["gas", "specific_angular_momentum_x"] / data["gas", "keplerian_specific_angular_momentum"]

@derived_field(name="normalized_specific_angular_momentum_y", sampling_type="cell", force_override=True)
def _normalized_specific_angular_momentum_y(field, data):
    return data["gas", "specific_angular_momentum_y"] / data["gas", "keplerian_specific_angular_momentum"]

@derived_field(name="normalized_specific_angular_momentum_z", sampling_type="cell", force_override=True)
def _normalized_specific_angular_momentum_z(field, data):
    return data["gas", "specific_angular_momentum_z"] / data["gas", "keplerian_specific_angular_momentum"]

@derived_field(name="keplerian_angular_momentum", sampling_type="cell", force_override=True)
def _keplerian_angular_momentum(field, data):
    return data["gas", "mass"] * data["gas", "keplerian_specific_angular_momentum"]

@derived_field(name="normalized_angular_momentum_x", sampling_type="cell", force_override=True)
def _normalized_angular_momentum_x(field, data):
    return data["gas", "angular_momentum_x"] / data["gas", "keplerian_angular_momentum"]

@derived_field(name="normalized_angular_momentum_y", sampling_type="cell", force_override=True)
def _normalized_angular_momentum_y(field, data):
    return data["gas", "angular_momentum_y"] / data["gas", "keplerian_angular_momentum"]

@derived_field(name="normalized_angular_momentum_z", sampling_type="cell", force_override=True)
def _normalized_angular_momentum_z(field, data):
    return data["gas", "angular_momentum_z"] / data["gas", "keplerian_angular_momentum"]

@derived_field(name="free_fall_time", sampling_type="cell", units="Myr", force_override=True)
def _free_fall_time(field, data):
    return np.pi / 2 * data["index","radius"]**(3/2) / np.sqrt(2 * phc.G * (SMBHMass * phc.msun + data["gas", "mass"]))

@derived_field(name="free_fall_ratio", sampling_type="cell", force_override=True)
def _free_fall_ratio(field, data):
    return data["gas", "cooling_time"] / data["gas", "free_fall_time"]

@derived_field(name="density_squared", sampling_type="cell", units="msun**2*Mpc**-6", force_override=True)
def _density_squared(field, data):
    return data["gas", "density"]**2

@derived_field(name="angle_theta", sampling_type="cell", force_override=True)
def _angle_theta(field, data):
    return np.arccos(data["index", "z"] / data["index", "radius"])

@derived_field(name="angle_phi", sampling_type="cell", force_override=True)
def _angle_phi(field, data):
    return np.arctan2(data["index", "y"], data["index", "x"])

@derived_field(name="vr", sampling_type="cell", units="km/s", force_override=True)
def _vr(field, data):
    return (data["gas", "velocity_x"] * data["index", "x"] + data["gas", "velocity_y"] * data["index", "y"] + data["gas", "velocity_z"] * data["index", "z"]) / data["index", "radius"]

@derived_field(name="vtheta", sampling_type="cell", units="1/s", force_override=True)
def _vtheta(field, data):
    return (data["gas", "vr"] * data["gas", "angle_theta"] - data["gas", "velocity_z"]) / (data["index", "radius"] * np.sin(data["gas", "angle_theta"]))

@derived_field(name="vphi", sampling_type="cell", units="km/s", force_override=True)
def _vphi(field, data):
    return data["gas", "velocity_y"] * np.cos(data["gas", "angle_phi"]) - data["gas", "velocity_x"] * np.sin(data["gas", "angle_phi"])

@derived_field(name="vtangent", sampling_type="cell", units="km/s", force_override=True)
def _vtangent(field, data):
    return np.sqrt(data["gas", "velocity_magnitude"]**2 - data["gas", "vr"]**2)

def makeFilename (pathName : str, baseExtension : str, n : int) -> str:
    if n < 10:
        file_n = '0000' + str(n)
    elif (n >= 10) & (n < 100):
        file_n = '000' + str(n)
    elif n >= 1000:
        file_n = '0' + str(n)
    else:
        file_n = '00' + str(n)

    return f"{pathName}{baseExtension}{file_n}.athdf"


def GetFlux(shell : yt.data_objects.selection_objects.cut_region.YTCutRegion):
    # flux_all = shell["gas", "vr"] * shell["gas", "density"]
    rv = shell["gas", "vr"]
    rho = shell["gas", "density"]
    flux_all = rv * rho
    temp = shell["gas", "temperature"].in_units("K")

    flux_in_cold = np.zeros_like(rv)
    flux_in_hot = np.zeros_like(rv)
    flux_out_cold = np.zeros_like(rv)
    flux_out_hot = np.zeros_like(rv)

    flux_in_cold = np.where((temp < 1.e6) & (rv < 0), -flux_all.in_units("Msun/pc**2/yr").value, 0.)
    flux_in_hot = np.where((temp >= 1.e6) & (rv < 0), -flux_all.in_units("Msun/pc**2/yr").value, 0.)

    flux_out_cold = np.where((temp < 1.e6) & (rv > 0), flux_all.in_units("Msun/pc**2/yr").value, 0.)
    flux_out_hot = np.where((temp >= 1.e6) & (rv > 0), flux_all.in_units("Msun/pc**2/yr").value, 0.)

    return [flux_in_cold, flux_in_hot, flux_out_cold, flux_out_hot]

def GetMassFlow(flux : np.ndarray, radius : float):
    return np.mean(flux) * 4 * np.pi * radius**2

def GetSphericalShell(ds, radius : float, width_tolerance : float, unit : str) -> yt.data_objects.selection_objects.cut_region.YTCutRegion:
    sphere = ds.sphere("c", (radius, unit))
    shell = sphere.cut_region([f"(obj['radius'].in_units('{unit}') > {radius - width_tolerance})"])
    
    return shell

def get_m_dot(location : str, base_ext : str, i : int, distances : np.ndarray):
    """From a dataset, extract the mass flow rates at different radii

    Args:
        location (str): path to the simulation data
        base_ext (str): base extension of the snapshot files
        i (int): snapshot number
        distances (np.ndarray): array of radii to extract mass flow rates

    Returns:
        tuple: (time, mass flow rates)
    """
    ds = yt.load(makeFilename(location, base_ext, i),units_override=units_override)
    time = float(ds.current_time.in_units("kyr").value)
    
    save_data = np.zeros((5, len(distances)))
    save_data[0] = distances

    for i, radius in enumerate(distances):
        shell = GetSphericalShell(ds, radius, radius * 0.07, "pc")
        flux_in_cold, flux_in_hot, flux_out_cold, flux_out_hot = GetFlux(shell)

        save_data[1,i] = GetMassFlow(flux_in_cold, radius)
        save_data[2,i] = GetMassFlow(flux_in_hot, radius)
        save_data[3,i] = GetMassFlow(flux_out_cold, radius)
        save_data[4,i] = GetMassFlow(flux_out_hot, radius)

    return (time, save_data)
    

def get_radial_data(location : str, base_ext : str, i : int, fields : list[str], maxR : float, minR : float, n_bins : int, units : list[str]):
    """From a dataset, extract the radial profiles

    Args:
        location (str): path to the simulation data
        base_ext (str): base extension of the snapshot files
        i (int): snapshot number
        fields (list[str]): list of fields to extract
        maxR (float): maximum radius to extract radial profiles, in pc
        minR (float): minimum radius to extract radial profiles, in pc
        n_bins (int): number of bins for the radial profiles
        units (list[str]): units of the fields
    Returns:
        tuple: (times, radial profiles)
    """

    ds = yt.load(makeFilename(location, base_ext, i),units_override=units_override)
    time = float(ds.current_time.in_units("kyr").value)

    sp0 = ds.sphere("c", maxR)
    rp0 = yt.create_profile(
        sp0,
        ("index", "radius"),
        fields,
        n_bins=n_bins,
        extrema={("index", "radius"): (minR, maxR)},
        units={("index", "radius"): "pc"},
        logs={("index", "radius"): True},
    )

    save_data = np.zeros((len(fields)+1, len(rp0.x)))
    save_data[0] = rp0.x.in_units("pc").value

    for i, field in enumerate(fields):
        save_data[i+1] = rp0[field].in_units(units[i]).value

    return (time, save_data)

def get_multiple_snapshots(location: str, base_ext: str, fields : list[str], start_nfile: int, stop_nfile: int, maxR: float, minR: float, n_bins: int, units: list[str]):
    """With multiprocessing, generate multiple snapshots of the simulation data at once and save them to the specified location

    Args:
        location (str): path to the simulation data
        base_ext (str): base extension of the snapshot files
        fields (list[str]): list of fields to extract
        start_nfile (int): starting snapshot number
        stop_nfile (int): ending snapshot number
        maxR (float): maximum radius to extract radial profiles, in pc
        minR (float): minimum radius to extract radial profiles, in pc
        n_bins (int): number of bins for the radial profiles
        units (list[str]): units of the fields
    Returns:
        dict: dictionary containing the radial profiles
    """
    print("Saving snapshots using {} cores".format(cpu_count()), flush=True)

    # keys = ['radius'] + fields
    total_data_save = {}
    # total_data_save['fields'] = keys

    with Pool() as p:
        items = [(location, base_ext, k, fields, maxR, minR, n_bins, units) for k in range(start_nfile, stop_nfile+1)]

        for k in enumerate(p.starmap(get_radial_data, items)):
            data = k[1][1]
            time = k[1][0]
            total_data_save[time] = data
            print(f"Snapshot {k[0]} done", flush=True)
    
    return total_data_save

def get_multiple_mdots(location: str, base_ext: str, start_nfile: int, stop_nfile: int, distances: np.ndarray):
    """With multiprocessing, generate multiple snapshots of the simulation data at once and save them to the specified location

    Args:
        location (str): path to the simulation data
        base_ext (str): base extension of the snapshot files
        start_nfile (int): starting snapshot number
        stop_nfile (int): ending snapshot number
        maxR (float): maximum radius to extract radial profiles, in pc
        minR (float): minimum radius to extract radial profiles, in pc
        n_bins (int): number of bins for the radial profiles
    Returns:
        dict: dictionary containing the radial profiles
    """

    number_of_processes = cpu_count()
    if number_of_processes > 30:
        print("More than 20 cores is available but might run out of memory", flush=True)
        number_of_processes = 30
        
    print("Saving snapshots using {} cores".format(number_of_processes), flush=True)

    total_data_save = {}

    with Pool(processes=number_of_processes) as p:
        items = [(location, base_ext, k, distances) for k in range(start_nfile, stop_nfile+1)]

        for k in enumerate(p.starmap(get_m_dot, items)):
            data = k[1][1]
            time = k[1][0]
            total_data_save[time] = data
            print(f"Snapshot {k[0]} done", flush=True)
    
    return total_data_save


def main():
    lookup_units = {'number_density': 'cm**-3', 
                    'pressure': 'Pa', 
                    'density': 'g/cm**3', 
                    'temperature': 'K', 
                    'normalized_angular_momentum_x':'',
                    'normalized_angular_momentum_y':'',
                    'normalized_angular_momentum_z':'', 
                    'vr':'km/s', 
                    'vtheta':'1/s', 
                    'vphi':'km/s', 
                    'vtangent':'km/s', 
                    'cooling_time':'Myr', 
                    'sound_crossing_time':'Myr', 
                    'cooling_rate':'erg/s/cm**3', 
                    'cooling_ratio':'',  
                    'normalized_specific_angular_momentum_x':'', 
                    'normalized_specific_angular_momentum_y':'', 
                    'normalized_specific_angular_momentum_z':'', 
                    'free_fall_time':'Myr', 
                    'free_fall_ratio':'', 
                    'specific_angular_momentum_x':'km**2/s', 
                    'specific_angular_momentum_y':'km**2/s', 
                    'specific_angular_momentum_z':'km**2/s', 
                    'angular_momentum_x':'km**2/s', 
                    'angular_momentum_y':'km**2/s', 
                    'angular_momentum_z':'km**2/s',
                    'm_dot_in_cold':'Msun/yr',
                    'm_dot_in_hot':'Msun/yr',
                    'm_dot_out_cold':'Msun/yr',
                    'm_dot_out_hot':'Msun/yr',}
    
    parser = argparse.ArgumentParser(description="Extract data from the output of the simulation")
    parser.add_argument("--path", dest="path", type=str, default=os.getcwd(), help="Path to the simulation output")
    parser.add_argument("--fields", dest="fields", type=str, nargs='+', action="store", help='Fields to extract')
    parser.add_argument('--base_ext', dest="base_ext", type=str, help='base extension of the simulation data')
    parser.add_argument('--snapshots', dest="snapshots", type=int, nargs='+', action="store", help='range of snapshots (start, end)')
    parser.add_argument('--maxR', dest="maxR", type=float, default=1.e5, help='Maximum radius to extract radial profiles, in pc', required=False)
    parser.add_argument('--minR', dest="minR", type=float, default=1., help='Minimum radius to extract radial profiles, in pc', required=False)

    args = parser.parse_args()

    path = args.path + '/'
    fields = args.fields
    base_ext = args.base_ext
    (start_nfile, stop_nfile) = args.snapshots
    maxR = args.maxR
    minR = args.minR
    n_bins = 80

    get_m_dot = False
    if "m_dot" in fields:
        get_m_dot = True
        fields.remove("m_dot")

    units_list = [lookup_units[field] for field in fields]

    print(f"Parameters: ")
    print(f"Path: {path}")
    print(f"Fields: {fields}")
    print(f"Base extension: {base_ext}")
    print(f"Snapshots: from {start_nfile} to {stop_nfile}")
    print(f"Units: {units_list}")
    print(f"Radii: from {minR} pc to {maxR} pc")

    total_data = get_multiple_snapshots(path, base_ext, fields, start_nfile, stop_nfile, maxR, minR, n_bins, units_list)
    sorted_dict = dict(sorted(total_data.items()))
    sorted_dict['fields'] = ['radius'] + fields

    # Save the data to a pickle file
    with open(f"{path}/radial_profiles.pkl", "wb") as f:
        pickle.dump(sorted_dict, f)
        print("Radial profiles saved to pickle file")

    if get_m_dot:
        distances = sorted_dict[list(sorted_dict.keys())[0]][0,:]
        print("Getting mass flow rates")
        m_dot_data = get_multiple_mdots(path, base_ext, start_nfile, stop_nfile, distances)
        sorted_m_dot = dict(sorted(m_dot_data.items()))
        sorted_m_dot['fields'] = ['radius', 'm_dot_in_cold', 'm_dot_in_hot', 'm_dot_out_cold', 'm_dot_out_hot']

        with open(f"{path}/mass_flow_rates.pkl", "wb") as f:
            pickle.dump(sorted_m_dot, f)
            print("Mass flow rates saved to pickle file")
    

if __name__ == "__main__":
    main()