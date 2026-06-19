#!/usr/bin/env python3
"""Lochhaas-style radial pressure-support decomposition.

This script follows the force-budget framework of Lochhaas et al. (2023):

    a_net = -1/rho dP_th/dr - 1/rho dP_turb/dr - 1/rho dP_ram/dr
            + v_rot**2/r - G M(<r)/r**2

The pressure-gradient terms are computed locally on a uniform Cartesian grid,
projected onto the radial direction, and then averaged in spherical shells by
summing force and dividing by shell gas mass. That is equivalent to a
mass-weighted average of the local accelerations.
"""

import argparse
import gc
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache")
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import numpy as np
import yt
import yt.units as u
from scipy.ndimage import gaussian_filter
from yt import physical_constants as phc

import athena_parameters as ap


MIN_RADIAL_CELLS = 3.0
# Match Lochhaas' 25 kpc smoothing scale in the largest default window, then
# use the same kernel size in cells for the smaller multiscale windows.
DEFAULT_DRIVING_CELLS = 25.0e3 * 384 / 1.408e5

DEFAULT_FIELDS = [
    "x",
    "y",
    "z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "pressure",
    "density",
    "temperature",
    "vr",
    "radius",
]

DEFAULT_MULTISCALE_SPECS = [
    {
        "name": "r000p01_000p23pc",
        "size_pc": 1.1,
        "r_min_pc": 1.6e-2,
        "r_max_pc": 5.0e-1,
        "nominal_min_pc": 1.0e-2,
        "nominal_max_pc": 2.3e-1,
        "dim": 384,
        "driving_pc": DEFAULT_DRIVING_CELLS * 1.1 / 384,
        "n_bins": 40,
    },
    {
        "name": "r000p23_005p3pc",
        "size_pc": 24.2,
        "r_min_pc": 1.9e-1,
        "r_max_pc": 11.0,
        "nominal_min_pc": 2.3e-1,
        "nominal_max_pc": 5.3,
        "dim": 384,
        "driving_pc": DEFAULT_DRIVING_CELLS * 24.2 / 384,
        "n_bins": 40,
    },
    {
        "name": "r005p3_121pc",
        "size_pc": 572.0,
        "r_min_pc": 4.5,
        "r_max_pc": 260.0,
        "nominal_min_pc": 5.3,
        "nominal_max_pc": 121.0,
        "dim": 384,
        "driving_pc": DEFAULT_DRIVING_CELLS * 572.0 / 384,
        "n_bins": 40,
    },
    {
        "name": "r121_2780pc",
        "size_pc": 1.32e4,
        "r_min_pc": 105.0,
        "r_max_pc": 6.0e3,
        "nominal_min_pc": 121.0,
        "nominal_max_pc": 2.78e3,
        "dim": 384,
        "driving_pc": DEFAULT_DRIVING_CELLS * 1.32e4 / 384,
        "n_bins": 40,
    },
    {
        "name": "r2780_64000pc",
        "size_pc": 1.408e5,
        "r_min_pc": 1.4e3,
        "r_max_pc": 6.4e4,
        "nominal_min_pc": 2.78e3,
        "nominal_max_pc": 6.4e4,
        "dim": 384,
        "driving_pc": DEFAULT_DRIVING_CELLS * 1.408e5 / 384,
        "n_bins": 40,
    },
]


def parse_args():
    """Parse and validate command-line options for single or multiscale runs."""

    parser = argparse.ArgumentParser(
        description="Compute Lochhaas-style pressure-support profiles."
    )
    parser.add_argument("--path", default=os.getcwd(), help="Snapshot directory.")
    parser.add_argument("--base-ext", default="M87.out2.", help="Snapshot basename.")
    parser.add_argument("--snapshot", type=int, required=True, help="Snapshot number.")
    parser.add_argument(
        "--size-kpc",
        type=float,
        default=10.0,
        help="Uniform-grid box width in kpc. Default: 10.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Uniform-grid cells per side. Memory scales as dim^3. Default: 256.",
    )
    parser.add_argument(
        "--driving-scale-kpc",
        type=float,
        default=10.0,
        help=(
            "Velocity smoothing/driving scale in kpc. For strict Lochhaas-style "
            "analysis, estimate this from a velocity structure function."
        ),
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=80,
        help="Number of radial profile bins. Default: 80.",
    )
    parser.add_argument(
        "--r-min-pc",
        type=float,
        default=None,
        help="Minimum profile radius in pc. Default: 3 grid cells.",
    )
    parser.add_argument(
        "--r-max-kpc",
        type=float,
        default=None,
        help="Maximum profile radius in kpc. Default: half the box width.",
    )
    parser.add_argument(
        "--density-max",
        type=float,
        default=None,
        help="Optional maximum gas density cut in g/cm^3.",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=None,
        help="Optional minimum gas temperature cut in K.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Default: pressure_support_lochhaas_<snapshot>.npz",
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help=(
            "Run five 384^3 centered cubes and stitch log-spaced radial ranges "
            "covering 0.01 pc to 64 kpc. The default turbulent driving scale "
            "is standardized in grid cells and anchored to 25 kpc in the "
            "largest window."
        ),
    )
    parser.add_argument(
        "--scale",
        action="append",
        default=None,
        metavar="NAME,SIZE_PC,RMIN_PC,RMAX_PC,DIM,DRIVING_PC,N_BINS",
        help=(
            "Override/add a multiscale cube. May be repeated. Example: "
            "inner,10,0.02,5,1024,1,80"
        ),
    )
    parser.add_argument(
        "--yt-log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="yt logger verbosity. Default: WARNING.",
    )
    args = parser.parse_args()

    if args.dim <= 0:
        parser.error("--dim must be positive.")
    if args.size_kpc <= 0:
        parser.error("--size-kpc must be positive.")
    if args.driving_scale_kpc <= 0:
        parser.error("--driving-scale-kpc must be positive.")
    if args.n_bins <= 0:
        parser.error("--n-bins must be positive.")
    if args.r_min_pc is not None and args.r_min_pc <= 0:
        parser.error("--r-min-pc must be positive.")
    if args.r_max_kpc is not None and args.r_max_kpc <= 0:
        parser.error("--r-max-kpc must be positive.")
    if args.scale is not None and not args.multiscale:
        parser.error("--scale requires --multiscale.")

    return args


def parse_scale_spec(spec):
    """Parse one comma-separated multiscale window override from the CLI."""

    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 7:
        raise ValueError(
            "Scale specs must have 7 comma-separated fields: "
            "NAME,SIZE_PC,RMIN_PC,RMAX_PC,DIM,DRIVING_PC,N_BINS"
        )

    name, size_pc, r_min_pc, r_max_pc, dim, driving_pc, n_bins = parts
    scale = {
        "name": name,
        "size_pc": float(size_pc),
        "r_min_pc": float(r_min_pc),
        "r_max_pc": float(r_max_pc),
        "dim": int(dim),
        "driving_pc": float(driving_pc),
        "n_bins": int(n_bins),
    }
    scale["nominal_min_pc"] = scale["r_min_pc"]
    scale["nominal_max_pc"] = scale["r_max_pc"]
    validate_scale_spec(scale)
    return scale


def validate_scale_spec(scale):
    """Check that one multiscale window has usable geometry and sampling."""

    if not scale["name"]:
        raise ValueError("Scale name cannot be empty.")
    if scale["size_pc"] <= 0:
        raise ValueError(f"Scale {scale['name']} has non-positive size.")
    if scale["r_min_pc"] <= 0:
        raise ValueError(f"Scale {scale['name']} has non-positive r_min.")
    if scale["r_max_pc"] <= scale["r_min_pc"]:
        raise ValueError(f"Scale {scale['name']} needs r_max > r_min.")
    if scale["r_max_pc"] > 0.5 * scale["size_pc"]:
        raise ValueError(
            f"Scale {scale['name']} has r_max larger than half the cube width."
        )
    if scale["dim"] <= 0:
        raise ValueError(f"Scale {scale['name']} has non-positive dim.")
    if scale["driving_pc"] <= 0:
        raise ValueError(f"Scale {scale['name']} has non-positive driving scale.")
    if scale["n_bins"] <= 0:
        raise ValueError(f"Scale {scale['name']} has non-positive n_bins.")


def get_scale_specs(args):
    """Return default or user-supplied multiscale window specifications."""

    if args.scale is None:
        return [dict(scale) for scale in DEFAULT_MULTISCALE_SPECS]
    return [parse_scale_spec(spec) for spec in args.scale]


def load_snapshot(path, base_ext, snapshot):
    """Load an Athena++ snapshot with the project unit overrides."""

    path = path.rstrip(os.sep) + os.sep
    filename = ap.makeFilename(path, base_ext, n=snapshot)
    return yt.load(filename, units_override=ap.units_override)


def smooth_quantity(quantity, sigma_cells, unit):
    """Gaussian-smooth a yt quantity after converting it to a target unit."""

    unit_obj = quantity.in_units(unit).units
    values = quantity.in_units(unit).value
    return gaussian_filter(values, sigma=sigma_cells) * unit_obj


def finite_quantity(quantity, unit):
    """Convert a quantity to a unit and replace non-finite values with NaN."""

    values = quantity.in_units(unit).value.copy()
    values[~np.isfinite(values)] = np.nan
    return values * quantity.in_units(unit).units


def radial_unit_vectors(datadict):
    """Build Cartesian components of the local radial unit vector."""

    x = datadict["x"].in_units("kpc").value
    y = datadict["y"].in_units("kpc").value
    z = datadict["z"].in_units("kpc").value
    radius = datadict["radius"].in_units("kpc").value

    r_hat_x = np.divide(x, radius, out=np.zeros_like(x), where=radius > 0)
    r_hat_y = np.divide(y, radius, out=np.zeros_like(y), where=radius > 0)
    r_hat_z = np.divide(z, radius, out=np.zeros_like(z), where=radius > 0)
    return r_hat_x, r_hat_y, r_hat_z


def radial_gradient(scalar_quantity, grid_resolution, r_hat, scalar_unit):
    """Compute a Cartesian gradient and project it along the radial direction."""

    scalar = scalar_quantity.in_units(scalar_unit)
    spacing_cm = grid_resolution.in_units("cm").value
    grad = np.gradient(scalar.value, spacing_cm, axis=(0, 1, 2))
    grad_x = grad[0] * scalar.units / u.cm
    grad_y = grad[1] * scalar.units / u.cm
    grad_z = grad[2] * scalar.units / u.cm
    r_hat_x, r_hat_y, r_hat_z = r_hat
    return grad_x * r_hat_x + grad_y * r_hat_y + grad_z * r_hat_z


def radial_velocity_gradient(velocity_quantity, grid_resolution, r_hat):
    """Compute the radial derivative of a velocity field in inverse seconds."""

    grad_v = radial_gradient(velocity_quantity, grid_resolution, r_hat, "cm/s")
    return grad_v.in_units("1/s")


def angular_velocity_components(datadict):
    """Convert Cartesian velocities into local spherical angular components."""

    x = datadict["x"].in_units("kpc").value
    y = datadict["y"].in_units("kpc").value
    z = datadict["z"].in_units("kpc").value
    radius = datadict["radius"].in_units("kpc").value
    r_cyl = np.sqrt(x**2 + y**2)

    cos_theta = np.divide(
        z,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0,
    )
    sin_theta = np.divide(
        r_cyl,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0,
    )
    cos_phi = np.divide(
        x,
        r_cyl,
        out=np.ones_like(r_cyl),
        where=r_cyl > 0,
    )
    sin_phi = np.divide(
        y,
        r_cyl,
        out=np.zeros_like(r_cyl),
        where=r_cyl > 0,
    )

    vx = datadict["velocity_x"].in_units("km/s")
    vy = datadict["velocity_y"].in_units("km/s")
    vz = datadict["velocity_z"].in_units("km/s")

    vtheta = (
        vx * cos_theta * cos_phi
        + vy * cos_theta * sin_phi
        - vz * sin_theta
    )
    vphi = vy * cos_phi - vx * sin_phi
    return vtheta.in_units("km/s"), vphi.in_units("km/s")


def enclosed_mass_total(radius):
    """Interpolate the total enclosed M87 mass at each radius."""

    if hasattr(ap, "enclosed_mass_at_radius"):
        return ap.enclosed_mass_at_radius(radius)

    radius_pc = radius.in_units("pc").value
    return np.interp(radius_pc, ap.radiusList, ap.massList) * u.Msun


def compute_support_accelerations(datadict, grid_resolution, driving_scale):
    """Compute local thermal, turbulent, ram, rotation, and gravity terms."""

    sigma_cells = float(driving_scale / grid_resolution / 6.0)
    r_hat = radial_unit_vectors(datadict)
    density = datadict["density"].in_units("g/cm**3")
    radius = datadict["radius"].in_units("cm")

    dPth_dr = radial_gradient(
        datadict["pressure"],
        grid_resolution,
        r_hat,
        "dyne/cm**2",
    )
    thermal = finite_quantity((-dPth_dr / density), "cm/s**2")

    vx = datadict["velocity_x"].in_units("km/s")
    vy = datadict["velocity_y"].in_units("km/s")
    vz = datadict["velocity_z"].in_units("km/s")
    vx_smooth = smooth_quantity(vx, sigma_cells, "km/s")
    vy_smooth = smooth_quantity(vy, sigma_cells, "km/s")
    vz_smooth = smooth_quantity(vz, sigma_cells, "km/s")
    density_smooth = smooth_quantity(density, sigma_cells, "g/cm**3")

    v_turb_sq = (
        (vx - vx_smooth) ** 2
        + (vy - vy_smooth) ** 2
        + (vz - vz_smooth) ** 2
    ) / 3.0
    turbulent_pressure = (density_smooth * v_turb_sq).in_units("dyne/cm**2")
    dPturb_dr = radial_gradient(
        turbulent_pressure,
        grid_resolution,
        r_hat,
        "dyne/cm**2",
    )
    turbulence = finite_quantity((-dPturb_dr / density), "cm/s**2")

    vr_smooth = smooth_quantity(datadict["vr"], sigma_cells, "km/s")
    dvr_dr = radial_velocity_gradient(vr_smooth, grid_resolution, r_hat)
    ram_pressure = (
        density_smooth * (dvr_dr * grid_resolution.in_units("cm")) ** 2
    ).in_units("dyne/cm**2")
    dPram_dr = radial_gradient(
        ram_pressure,
        grid_resolution,
        r_hat,
        "dyne/cm**2",
    )
    ram_lochhaas = finite_quantity((-dPram_dr / density), "cm/s**2")

    radial_inertia = finite_quantity(
        (-(vr_smooth.in_units("cm/s") * dvr_dr)).in_units("cm/s**2"),
        "cm/s**2",
    )

    vtheta, vphi = angular_velocity_components(datadict)
    vtheta_smooth = smooth_quantity(vtheta, sigma_cells, "km/s")
    vphi_smooth = smooth_quantity(vphi, sigma_cells, "km/s")
    rotation = finite_quantity(
        (
            (
                vtheta_smooth.in_units("cm/s") ** 2
                + vphi_smooth.in_units("cm/s") ** 2
            )
            / radius
        ),
        "cm/s**2",
    )

    enclosed_mass = enclosed_mass_total(datadict["radius"])
    gravity_magnitude = finite_quantity(
        (phc.G * enclosed_mass / radius**2),
        "cm/s**2",
    )

    support_terms = {
        "thermal": thermal,
        "turbulence": turbulence,
        "ram_lochhaas": ram_lochhaas,
        "rotation": rotation,
    }
    diagnostics = {
        "radial_inertia": radial_inertia,
        "gravity_magnitude": gravity_magnitude,
        "turbulent_pressure": turbulent_pressure,
        "ram_pressure_lochhaas": ram_pressure,
    }

    return support_terms, diagnostics, sigma_cells


def make_radial_bins(grid_resolution, box_width, n_bins, r_min_pc=None, r_max_kpc=None):
    """Create logarithmic spherical-shell edges for one uniform-grid cube."""

    if r_min_pc is None:
        r_min = (
            max(MIN_RADIAL_CELLS * grid_resolution.in_units("pc").value, 1e-2)
            * u.pc
        )
    else:
        r_min = r_min_pc * u.pc

    if r_max_kpc is None:
        r_max = 0.5 * box_width
    else:
        r_max = r_max_kpc * u.kpc

    if r_max <= r_min:
        raise ValueError("Maximum radius must be larger than minimum radius.")

    return (
        np.logspace(
            np.log10(r_min.in_units("pc").value),
            np.log10(r_max.in_units("pc").value),
            n_bins + 1,
        )
        * u.pc
    )


def shell_mass_weighted_profiles(
    acceleration_terms,
    diagnostics,
    radius,
    density,
    grid_resolution,
    radial_bins,
    base_mask=None,
):
    """Mass-weight local accelerations and convert them into support profiles."""

    radius_pc = radius.in_units("pc").value
    density_cgs = density.in_units("g/cm**3").value
    cell_volume = grid_resolution.in_units("cm").value**3
    cell_mass = density_cgs * cell_volume

    if base_mask is None:
        base_mask = np.ones_like(radius_pc, dtype=bool)

    keys = list(acceleration_terms) + ["radial_inertia", "gravity_magnitude"]
    all_terms = dict(acceleration_terms)
    all_terms["radial_inertia"] = diagnostics["radial_inertia"]
    all_terms["gravity_magnitude"] = diagnostics["gravity_magnitude"]

    profiles = {
        "r_pc": np.sqrt(radial_bins[:-1] * radial_bins[1:]).in_units("pc").value,
        "r_inner_pc": radial_bins[:-1].in_units("pc").value,
        "r_outer_pc": radial_bins[1:].in_units("pc").value,
        "shell_mass_g": np.full(len(radial_bins) - 1, np.nan),
        "cell_count": np.zeros(len(radial_bins) - 1, dtype=int),
    }

    for key in keys:
        profiles[f"a_{key}_cm_s2"] = np.full(len(radial_bins) - 1, np.nan)

    for i in range(len(radial_bins) - 1):
        shell_mask = (
            base_mask
            & (radius_pc >= radial_bins[i].in_units("pc").value)
            & (radius_pc < radial_bins[i + 1].in_units("pc").value)
            & np.isfinite(cell_mass)
            & (cell_mass > 0)
        )
        if not np.any(shell_mask):
            continue

        mass = cell_mass[shell_mask]
        profiles["shell_mass_g"][i] = np.sum(mass)
        profiles["cell_count"][i] = np.count_nonzero(shell_mask)

        for key, values in all_terms.items():
            accel = values.in_units("cm/s**2").value
            valid = shell_mask & np.isfinite(accel)
            if not np.any(valid):
                continue
            profiles[f"a_{key}_cm_s2"][i] = (
                np.sum(cell_mass[valid] * accel[valid]) / np.sum(cell_mass[valid])
            )

    gravity = profiles["a_gravity_magnitude_cm_s2"]
    support_valid = np.isfinite(gravity) & (gravity > 0)
    total_support = np.zeros_like(gravity)

    for key in acceleration_terms:
        accel = profiles[f"a_{key}_cm_s2"]
        profiles[f"support_{key}"] = np.divide(
            accel,
            gravity,
            out=np.full_like(accel, np.nan, dtype=float),
            where=support_valid,
        )
        total_support += np.where(
            support_valid & np.isfinite(profiles[f"support_{key}"]),
            profiles[f"support_{key}"],
            0.0,
        )

    total_support[~support_valid] = np.nan
    profiles["support_total_non_gravity"] = total_support
    no_ram_keys = [key for key in acceleration_terms if key != "ram_lochhaas"]
    total_support_no_ram = np.zeros_like(gravity)

    for key in no_ram_keys:
        total_support_no_ram += np.where(
            support_valid & np.isfinite(profiles[f"support_{key}"]),
            profiles[f"support_{key}"],
            0.0,
        )

    total_support_no_ram[~support_valid] = np.nan
    profiles["support_total_non_gravity_no_ram"] = total_support_no_ram
    profiles["support_radial_inertia"] = np.divide(
        profiles["a_radial_inertia_cm_s2"],
        gravity,
        out=np.full_like(gravity, np.nan, dtype=float),
        where=support_valid,
    )
    profiles["a_net_without_inertia_cm_s2"] = (
        profiles["a_thermal_cm_s2"]
        + profiles["a_turbulence_cm_s2"]
        + profiles["a_ram_lochhaas_cm_s2"]
        + profiles["a_rotation_cm_s2"]
        - gravity
    )
    profiles["a_net_without_ram_or_inertia_cm_s2"] = (
        profiles["a_thermal_cm_s2"]
        + profiles["a_turbulence_cm_s2"]
        + profiles["a_rotation_cm_s2"]
        - gravity
    )
    profiles["a_net_with_inertia_cm_s2"] = (
        profiles["a_net_without_inertia_cm_s2"]
        + profiles["a_radial_inertia_cm_s2"]
    )

    return profiles


def build_mask(datadict, density_max=None, temperature_min=None):
    """Build the finite gas-cell mask, with optional density/temperature cuts."""

    density = datadict["density"].in_units("g/cm**3")
    temperature = datadict["temperature"].in_units("K")
    radius = datadict["radius"].in_units("pc")

    mask = np.isfinite(radius.value) & (radius.value > 0)
    mask &= np.isfinite(density.value) & (density.value > 0)

    if density_max is not None:
        mask &= density.value < density_max
    if temperature_min is not None:
        mask &= temperature.value >= temperature_min

    return mask


def save_profiles(output_path, profiles, metadata):
    """Write profile arrays and stringified metadata to an NPZ file."""

    save_dict = dict(profiles)
    save_dict["metadata_keys"] = np.array(list(metadata.keys()), dtype=str)
    save_dict["metadata_values"] = np.array(
        [str(value) for value in metadata.values()],
        dtype=str,
    )
    np.savez(output_path, **save_dict)


def run_profile_for_cube(
    ds,
    size_pc,
    dim,
    driving_pc,
    n_bins,
    r_min_pc,
    r_max_pc,
    density_max=None,
    temperature_min=None,
):
    """Regrid one cube, compute local terms, and shell-average the profile."""

    box_width = size_pc * u.pc
    grid_resolution = box_width / dim
    driving_scale = driving_pc * u.pc

    effective_min_pc = MIN_RADIAL_CELLS * grid_resolution.in_units("pc").value
    if r_min_pc < effective_min_pc:
        print(
            "Warning: requested r_min "
            f"{r_min_pc:.4g} pc is below {MIN_RADIAL_CELLS:g} grid cells "
            f"({effective_min_pc:.4g} pc) for dim={dim}, size={size_pc:g} pc. "
            "The profile will be computed, but the innermost gradients are "
            "resolution-sensitive.",
            flush=True,
        )

    datadict, time_kyr = ap.regrid_yt(
        ds,
        fields=DEFAULT_FIELDS,
        dim=[dim, dim, dim],
        bounding_length=[
            box_width.in_units("kpc").value,
            box_width.in_units("kpc").value,
            box_width.in_units("kpc").value,
        ],
    )

    support_terms, diagnostics, sigma_cells = compute_support_accelerations(
        datadict,
        grid_resolution,
        driving_scale,
    )

    radial_bins = make_radial_bins(
        grid_resolution,
        box_width,
        n_bins,
        r_min_pc=r_min_pc,
        r_max_kpc=r_max_pc / 1.0e3,
    )
    base_mask = build_mask(
        datadict,
        density_max=density_max,
        temperature_min=temperature_min,
    )

    profiles = shell_mass_weighted_profiles(
        support_terms,
        diagnostics,
        datadict["radius"],
        datadict["density"],
        grid_resolution,
        radial_bins,
        base_mask=base_mask,
    )

    del datadict, support_terms, diagnostics
    gc.collect()

    cube_metadata = {
        "time_kyr": time_kyr,
        "size_pc": size_pc,
        "dim": dim,
        "grid_resolution_pc": grid_resolution.in_units("pc").value,
        "driving_pc": driving_pc,
        "sigma_cells": sigma_cells,
        "r_min_pc": r_min_pc,
        "r_max_pc": r_max_pc,
        "n_bins": n_bins,
    }
    return profiles, cube_metadata


def scale_blend_weight(r_pc, scale):
    """Return log-radius taper weights for one multiscale cube."""

    r_pc = np.asarray(r_pc)
    weight = np.zeros_like(r_pc, dtype=float)

    r_min = scale["r_min_pc"]
    r_max = scale["r_max_pc"]
    nominal_min = scale.get("nominal_min_pc", r_min)
    nominal_max = scale.get("nominal_max_pc", r_max)

    in_scale = (r_pc >= r_min) & (r_pc <= r_max)
    in_nominal = in_scale & (r_pc >= nominal_min) & (r_pc <= nominal_max)
    weight[in_nominal] = 1.0

    log_r = np.log10(r_pc, where=r_pc > 0, out=np.full_like(r_pc, np.nan))

    if nominal_min > r_min:
        low = in_scale & (r_pc < nominal_min)
        denom = np.log10(nominal_min) - np.log10(r_min)
        weight[low] = (log_r[low] - np.log10(r_min)) / denom

    if nominal_max < r_max:
        high = in_scale & (r_pc > nominal_max)
        denom = np.log10(r_max) - np.log10(nominal_max)
        weight[high] = 1.0 - (log_r[high] - np.log10(nominal_max)) / denom

    return np.clip(weight, 0.0, 1.0)


def combine_multiscale_profiles(profile_items, scale_specs):
    """Interpolate and blend per-cube profiles onto one radial grid."""

    first_profiles = profile_items[0][1]
    numeric_keys = [
        key
        for key in first_profiles
        if np.asarray(first_profiles[key]).ndim == 1
        and key not in {"r_pc", "r_inner_pc", "r_outer_pc"}
    ]

    r_min = min(scale.get("nominal_min_pc", scale["r_min_pc"]) for scale in scale_specs)
    r_max = max(scale.get("nominal_max_pc", scale["r_max_pc"]) for scale in scale_specs)
    n_bins = sum(scale["n_bins"] for scale in scale_specs)
    radial_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    r_pc = np.sqrt(radial_edges[:-1] * radial_edges[1:])

    combined = {
        "r_pc": r_pc,
        "r_inner_pc": radial_edges[:-1],
        "r_outer_pc": radial_edges[1:],
    }
    weight_sum = np.zeros_like(r_pc, dtype=float)
    scale_weights = []

    profile_by_name = {name: profiles for name, profiles, _ in profile_items}

    for scale in scale_specs:
        weight = scale_blend_weight(r_pc, scale)
        scale_weights.append(weight)
        weight_sum += weight

    for key in numeric_keys:
        numerator = np.zeros_like(r_pc, dtype=float)
        denominator = np.zeros_like(r_pc, dtype=float)

        for scale, weight in zip(scale_specs, scale_weights):
            profiles = profile_by_name[scale["name"]]
            source_r = np.asarray(profiles["r_pc"])
            source_y = np.asarray(profiles[key])
            valid = np.isfinite(source_r) & np.isfinite(source_y) & (source_r > 0)
            if np.count_nonzero(valid) < 2:
                continue

            interpolated = np.interp(
                np.log10(r_pc),
                np.log10(source_r[valid]),
                source_y[valid],
                left=np.nan,
                right=np.nan,
            )
            valid_interp = np.isfinite(interpolated) & (weight > 0)
            numerator[valid_interp] += weight[valid_interp] * interpolated[valid_interp]
            denominator[valid_interp] += weight[valid_interp]

        combined[key] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=denominator > 0,
        )

    scale_names = np.array([scale["name"] for scale in scale_specs], dtype="<U32")
    scale_weight_matrix = np.vstack(scale_weights)
    best_scale = np.argmax(scale_weight_matrix, axis=0)
    combined["scale_name"] = scale_names[best_scale]
    combined["scale_blend_weight_sum"] = weight_sum

    return combined


def run_single_scale(args, ds):
    """Run the pressure-support calculation for one centered uniform cube."""

    size_pc = args.size_kpc * 1.0e3
    r_max_pc = args.r_max_kpc * 1.0e3 if args.r_max_kpc is not None else 0.5 * size_pc
    r_min_pc = args.r_min_pc

    if r_min_pc is None:
        grid_resolution_pc = size_pc / args.dim
        r_min_pc = max(MIN_RADIAL_CELLS * grid_resolution_pc, 1e-2)

    print(f"Uniform grid: {args.dim}^3")
    print(f"Grid width: {size_pc:.4g} pc")
    print(f"Requested radial range: {r_min_pc:.4g}-{r_max_pc:.4g} pc")
    print(f"Driving scale: {args.driving_scale_kpc:.4g} kpc")

    profiles, cube_metadata = run_profile_for_cube(
        ds,
        size_pc=size_pc,
        dim=args.dim,
        driving_pc=args.driving_scale_kpc * 1.0e3,
        n_bins=args.n_bins,
        r_min_pc=r_min_pc,
        r_max_pc=r_max_pc,
        density_max=args.density_max,
        temperature_min=args.temperature_min,
    )
    return profiles, cube_metadata


def run_multiscale(args, ds):
    """Run all multiscale cubes and stitch their profiles together."""

    profile_items = []
    scale_metadata = []
    transition_pc = []
    window_size_pc = []

    scale_specs = get_scale_specs(args)
    for i, scale in enumerate(scale_specs):
        print(
            "\n"
            f"Running scale '{scale['name']}': "
            f"cube={scale['size_pc']:.4g} pc, "
            f"r={scale['r_min_pc']:.4g}-{scale['r_max_pc']:.4g} pc, "
            f"dim={scale['dim']}, "
            f"driving={scale['driving_pc']:.4g} pc",
            flush=True,
        )
        profiles, cube_metadata = run_profile_for_cube(
            ds,
            size_pc=scale["size_pc"],
            dim=scale["dim"],
            driving_pc=scale["driving_pc"],
            n_bins=scale["n_bins"],
            r_min_pc=scale["r_min_pc"],
            r_max_pc=scale["r_max_pc"],
            density_max=args.density_max,
            temperature_min=args.temperature_min,
        )
        profile_items.append((scale["name"], profiles, cube_metadata))
        scale_metadata.append((scale["name"], cube_metadata))
        window_size_pc.append(scale["size_pc"])
        if i > 0:
            transition_pc.append(scale.get("nominal_min_pc", scale["r_min_pc"]))

    profiles = combine_multiscale_profiles(profile_items, scale_specs)
    profiles["scale_transition_pc"] = np.array(transition_pc)
    profiles["scale_window_size_pc"] = np.array(window_size_pc)
    profiles["scale_window_name"] = np.array(
        [scale["name"] for scale in scale_specs],
        dtype="<U32",
    )
    return profiles, scale_metadata


def main():
    """CLI entry point: load data, run profiles, and save the NPZ output."""

    args = parse_args()
    yt.funcs.mylog.setLevel(args.yt_log_level)

    print(f"Loading snapshot {args.snapshot} from {args.path}")
    ds = load_snapshot(args.path, args.base_ext, args.snapshot)

    if args.multiscale:
        profiles, scale_metadata = run_multiscale(args, ds)
        mode_metadata = {
            "mode": "multiscale",
            "scale_metadata": scale_metadata,
        }
    else:
        profiles, cube_metadata = run_single_scale(args, ds)
        mode_metadata = {
            "mode": "single",
            "cube_metadata": cube_metadata,
        }

    output = args.output
    if output is None:
        if args.multiscale:
            output = f"pressure_support_lochhaas_multiscale_{args.snapshot:05d}.npz"
        else:
            output = f"pressure_support_lochhaas_{args.snapshot:05d}.npz"

    metadata = {
        "snapshot": args.snapshot,
        "density_max_g_cm3": args.density_max,
        "temperature_min_K": args.temperature_min,
        "ram_definition": "rho_smooth * (dvr_smooth_dr * dx)^2",
        "rotation_definition": (
            "convert Cartesian velocity to spherical vtheta/vphi on the "
            "uniform grid, Gaussian smooth those angular velocity fields, "
            "then compute (vtheta_sm^2 + vphi_sm^2) / r with no radial gradient"
        ),
        "averaging": "sum(cell_mass * acceleration) / sum(cell_mass)",
    }
    metadata.update(mode_metadata)
    save_profiles(output, profiles, metadata)

    print(f"Wrote {output}")
    print(
        "Use support_total_non_gravity for the signed Lochhaas support ratio; "
        "support_radial_inertia is a diagnostic and is not included in that total."
    )


if __name__ == "__main__":
    main()
