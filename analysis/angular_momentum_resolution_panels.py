#!/usr/bin/env python3
"""Build the resolution comparison morphology / angular-momentum figure."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import yt
import yt.units as u

import athena_parameters as ap


DEFAULT_OUTPUT_PREFIX = Path("angular_momentum_resolution_panels")
DEFAULT_CACHE = Path("angular_momentum_resolution_panels_922myr_coolgas_cache.npz")
ALL_GAS_OUTPUT_PREFIX = Path("angular_momentum_resolution_panels_allgas_projection")
ALL_GAS_CACHE = Path("angular_momentum_resolution_panels_922myr_allgas_cache.npz")
BASE_EXT = "M87.out2."
WIDTH_PC = 800.0
SCALE_PC = 200.0
R_ACC_R8_PC = 4**8 * 1.0e-3
R_ACC_R10_PC = 4**7 * 1.0e-3
PROFILE_RADIUS_PC = 65.0
DENSITY_VMIN = 2.0e-1
DENSITY_VMAX = 3.0e3

FIELD_X = "normalized_specific_angular_momentum_x_cold"
FIELD_Y = "normalized_specific_angular_momentum_y_cold"
FIELD_Z = "normalized_specific_angular_momentum_z_cold"

OKABE_ITO = {
    "x": "#0072B2",
    "y": "#E69F00",
    "z": "#009E73",
    "total": "black",
}
MUTED = {
    "x": "#4C78A8",
    "y": "#B65C41",
    "z": "#2A9D8F",
    "total": "#222222",
}

MORPHOLOGY_PANELS = [
    {
        "key": "a",
        "run": "R8",
        "time_label": "920 Myr",
        "snapshot": 920,
        "path": Path("/Users/tvh0021/Documents/m87-pleiades/8smr-f0.8/rf920Myr-fine"),
        "r_acc_pc": R_ACC_R8_PC,
    },
    {
        "key": "b",
        "run": "R8",
        "time_label": "922 Myr",
        "snapshot": 928,
        "path": Path("/Users/tvh0021/Documents/m87-pleiades/8smr-f0.8/rf920Myr-fine"),
        "r_acc_pc": R_ACC_R8_PC,
    },
    {
        "key": "c",
        "run": "R10",
        "time_label": "922 Myr",
        "snapshot": 928,
        "path": Path("/Users/tvh0021/Documents/m87-pleiades/10smr-f0.8/rf920Myr"),
        "r_acc_pc": R_ACC_R10_PC,
    },
]


def register_colormaps() -> None:
    ehtplot_path = Path("/Users/tvh0021/git_repos/ehtplot")
    if ehtplot_path.exists() and str(ehtplot_path) not in sys.path:
        sys.path.append(str(ehtplot_path))

    try:
        import ehtplot.color  # noqa: F401
    except Exception as exc:
        print(f"Warning: could not register ehtplot colormaps ({exc}).")


def text_effect(linewidth: float = 2.8, color: str = "black", alpha: float = 0.75):
    return [path_effects.withStroke(linewidth=linewidth, foreground=color, alpha=alpha)]


def add_panel_label(ax, letter: str, *, color: str = "white", stroke: str | None = "black") -> None:
    label = ax.text(
        0.035,
        0.04,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=color,
        fontsize=30,
        fontweight="bold",
        zorder=30,
    )
    if stroke is not None:
        label.set_path_effects(text_effect(color=stroke))


def add_scale_bar(ax) -> None:
    half = 0.5 * WIDTH_PC
    x0 = -0.30 * WIDTH_PC
    x1 = x0 + SCALE_PC
    y = -0.385 * WIDTH_PC
    ax.plot([x0, x1], [y, y], color="black", lw=2.2, solid_capstyle="butt", zorder=25)
    ax.text(
        0.5 * (x0 + x1),
        y - 0.035 * WIDTH_PC,
        "200 pc",
        ha="center",
        va="top",
        color="black",
        fontsize=22,
        zorder=26,
    )
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)


def snapshot_path(panel: dict) -> Path:
    return panel["path"] / f"{BASE_EXT}{panel['snapshot']:05d}.athdf"


def load_morphology_panel(panel: dict, *, buff_size: int, morphology_gas: str) -> dict[str, np.ndarray | float | str]:
    path = snapshot_path(panel)
    print(f"Loading morphology snapshot: {path}")
    ds = yt.load(str(path), units_override=ap.units_override)

    r_in = 4**2 * u.mpc
    sphere = ds.sphere("c", (WIDTH_PC, "pc")).cut_region(
        [
            f"obj['radius'].in_units('pc') > {r_in.in_units('pc').value}",
            f"obj['radius'].in_units('pc') < {WIDTH_PC}",
        ]
    )
    if morphology_gas == "cool":
        projection_source = sphere.cut_region(
            ["obj['temperature'].in_units('K') < 1.e6", "obj['vr'].in_units('km/s') < 1.e3"]
        )
    elif morphology_gas == "all":
        projection_source = sphere
    else:
        raise ValueError(f"Unsupported morphology_gas={morphology_gas!r}")

    proj = yt.ProjectionPlot(
        ds,
        "z",
        ("gas", "number_density"),
        width=(WIDTH_PC, "pc"),
        buff_size=(buff_size, buff_size),
        weight_field=("gas", "density"),
        data_source=projection_source,
    )
    proj.set_unit(("gas", "number_density"), "1/cm**3")
    density = np.asarray(proj.frb[("gas", "number_density")])

    vel_x = yt.ProjectionPlot(
        ds,
        "z",
        ("gas", "velocity_x"),
        width=(WIDTH_PC, "pc"),
        buff_size=(buff_size, buff_size),
        weight_field=("gas", "density"),
        data_source=projection_source,
    )
    vel_y = yt.ProjectionPlot(
        ds,
        "z",
        ("gas", "velocity_y"),
        width=(WIDTH_PC, "pc"),
        buff_size=(buff_size, buff_size),
        weight_field=("gas", "density"),
        data_source=projection_source,
    )

    return {
        "density": density,
        "vx": np.asarray(vel_x.frb[("gas", "velocity_x")].in_units("km/s")),
        "vy": np.asarray(vel_y.frb[("gas", "velocity_y")].in_units("km/s")),
        "current_time_myr": float(ds.current_time.in_units("Myr")),
        "r_acc_pc": float(panel["r_acc_pc"]),
        "run": panel["run"],
        "time_label": panel["time_label"],
    }


def load_or_compute_morphology(
    cache_path: Path,
    *,
    recompute: bool,
    use_cache: bool,
    buff_size: int,
    morphology_gas: str,
) -> dict[str, dict]:
    if use_cache and cache_path.exists() and not recompute:
        print(f"Loading cached morphology data: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            data = {}
            for panel in MORPHOLOGY_PANELS:
                key = panel["key"]
                data[key] = {
                    "density": cache[f"{key}_density"],
                    "vx": cache[f"{key}_vx"],
                    "vy": cache[f"{key}_vy"],
                    "current_time_myr": float(cache[f"{key}_current_time_myr"]),
                    "r_acc_pc": float(cache[f"{key}_r_acc_pc"]),
                    "run": str(cache[f"{key}_run"]),
                    "time_label": str(cache[f"{key}_time_label"]),
                }
            return data

    data = {
        panel["key"]: load_morphology_panel(panel, buff_size=buff_size, morphology_gas=morphology_gas)
        for panel in MORPHOLOGY_PANELS
    }
    if use_cache:
        payload = {}
        for key, panel_data in data.items():
            for name, value in panel_data.items():
                payload[f"{key}_{name}"] = np.asarray(value)
        np.savez_compressed(cache_path, **payload)
        print(f"Saved morphology cache: {cache_path}")
    return data


def load_profile_data(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    fields = list(raw.pop("fields"))
    times = sorted(raw)
    radii = np.asarray(raw[times[0]][0], dtype=float)
    data = np.empty((len(times), len(fields), len(radii)), dtype=float)
    for i, time in enumerate(times):
        data[i] = raw[time]
    return fields, np.asarray(times, dtype=float) / 1.0e3, radii, data


def load_angular_momentum_data() -> dict[str, dict]:
    runs = {
        "R8": Path("/Users/tvh0021/Documents/m87-pleiades/8smr-f0.8/rf920Myr-fine/radial_profiles_wmass.pkl"),
        "R10": Path("/Users/tvh0021/Documents/m87-pleiades/10smr-f0.8/rf920Myr/radial_profiles_wmass.pkl"),
    }
    result = {}
    for run, path in runs.items():
        fields, times_myr, radii_pc, data = load_profile_data(path)
        mask = (times_myr >= 920.0) & (times_myr <= 1020.0)
        idx = int(np.abs(radii_pc - PROFILE_RADIUS_PC).argmin())
        lx = data[mask, fields.index(FIELD_X), idx]
        ly = data[mask, fields.index(FIELD_Y), idx]
        lz = data[mask, fields.index(FIELD_Z), idx]
        result[run] = {
            "time_myr": times_myr[mask],
            "radius_pc": float(radii_pc[idx]),
            "lx": lx,
            "ly": ly,
            "lz": lz,
            "total": np.sqrt(lx**2 + ly**2 + lz**2),
        }
    return result


def plot_morphology(ax, data: dict, *, show_colorbar: bool = False):
    half = 0.5 * WIDTH_PC
    extent = (-half, half, -half, half)
    cmap = plt.get_cmap("afmhot_10us" if "afmhot_10us" in plt.colormaps() else "afmhot").copy()
    cmap.set_bad("white")
    density = np.asarray(data["density"], dtype=float)
    density = np.where(density > 0.0, density, np.nan)
    image = ax.imshow(
        density,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX),
        interpolation="bicubic",
        rasterized=True,
    )

    stride = max(1, density.shape[0] // 80)
    coord = np.linspace(-half, half, density.shape[0])[::stride]
    ax.streamplot(
        coord,
        coord,
        data["vx"][::stride, ::stride],
        data["vy"][::stride, ::stride],
        color="#0072B2",
        density=2.0,
        linewidth=0.65,
        arrowsize=0.85,
        minlength=0.05,
        maxlength=4.0,
        broken_streamlines=True,
        zorder=10,
    )

    circle = plt.Circle(
        (0.0, 0.0),
        float(data["r_acc_pc"]),
        fill=False,
        color="#009E73",
        ls="--",
        lw=2.2,
        zorder=18,
    )
    ax.add_patch(circle)
    add_scale_bar(ax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.05, 0.94, data["run"], transform=ax.transAxes, ha="left", va="top", fontsize=22, color="black", zorder=25)
    ax.text(0.95, 0.94, data["time_label"], transform=ax.transAxes, ha="right", va="top", fontsize=22, color="black", zorder=25)
    return image if show_colorbar else None


def plot_angular_momentum(ax, series: dict, palette: dict, *, ylabel: bool, label: str) -> None:
    time = series["time_myr"]
    ax.plot(time, series["lx"], color=palette["x"], ls="--", lw=2.0, label=r"$L_x$")
    ax.plot(time, series["ly"], color=palette["y"], ls="--", lw=2.0, label=r"$L_y$")
    ax.plot(time, series["lz"], color=palette["z"], ls="--", lw=2.0, label=r"$L_z$")
    ax.plot(time, series["total"], color=palette["total"], lw=2.0, label=r"$|\mathbf{L}|$")
    ax.set_xlim(918.0, 1022.0)
    ax.set_ylim(-1.1, 1.3)
    ax.set_xlabel("Time (Myr)", fontsize=22)
    if ylabel:
        ax.set_ylabel(r"$L/L_{\rm K}$ ($r=R_{\rm acc,R8}$)", fontsize=22)
    ax.grid(True, color="0.7", alpha=0.55, lw=0.8)
    ax.text(0.05, 0.97, label, transform=ax.transAxes, ha="left", va="top", fontsize=22, color="black")
    ax.tick_params(axis="both", which="major", labelsize=22, length=5)
    ax.tick_params(axis="both", which="minor", length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def build_figure(
    morphology: dict[str, dict],
    angular_momentum: dict[str, dict],
    *,
    output_prefix: Path,
    palette: dict,
    save_pdf: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 22,
            "savefig.bbox": "tight",
        }
    )

    fig = plt.figure(figsize=(14.6, 9.8), dpi=220)
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.0675],
        height_ratios=[1.0, 0.9405],
        left=0.045,
        right=0.965,
        bottom=0.07,
        top=0.97,
        wspace=0.08,
        hspace=0.06,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])
    ax_note = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2], sharey=ax_d)

    image = None
    for ax, key in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")]:
        image_candidate = plot_morphology(ax, morphology[key], show_colorbar=(key == "c"))
        image = image_candidate or image
        add_panel_label(ax, key, color="black", stroke="white")

    fig.canvas.draw()
    cbar_pos = cax.get_position()
    panel_pos = ax_c.get_position()
    cax.set_position([cbar_pos.x0, panel_pos.y0, cbar_pos.width, panel_pos.height])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(r"$n$ (cm$^{-3}$)", fontsize=22)
    cbar.ax.tick_params(labelsize=22, length=4)

    ax_note.axis("off")

    plot_angular_momentum(ax_d, angular_momentum["R8"], palette, ylabel=True, label="R8")
    plot_angular_momentum(ax_e, angular_momentum["R10"], palette, ylabel=False, label="R10")
    plt.setp(ax_e.get_yticklabels(), visible=False)
    add_panel_label(ax_d, "d", color="black", stroke="white")
    add_panel_label(ax_e, "e", color="black", stroke="white")

    handles, labels = ax_e.get_legend_handles_labels()
    ax_e.legend(
        handles,
        labels,
        loc="lower right",
        ncol=2,
        frameon=True,
        fontsize=20,
        handlelength=1.55,
        borderpad=0.25,
        labelspacing=0.15,
        columnspacing=0.65,
        borderaxespad=0.25,
    )

    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    if save_pdf:
        fig.savefig(output_prefix.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--buff-size", type=int, default=900)
    parser.add_argument(
        "--morphology-gas",
        choices=("cool", "all"),
        default="cool",
        help="Gas selection used only for the top morphology projections.",
    )
    parser.add_argument("--recompute-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--skip-muted-preview",
        action="store_true",
        help="Write only the primary Okabe-Ito version.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_colormaps()

    cache_path = args.cache.expanduser()
    output_prefix = args.output_prefix.expanduser()
    if args.morphology_gas == "all":
        if args.cache == DEFAULT_CACHE:
            cache_path = ALL_GAS_CACHE
        if args.output_prefix == DEFAULT_OUTPUT_PREFIX:
            output_prefix = ALL_GAS_OUTPUT_PREFIX

    morphology = load_or_compute_morphology(
        cache_path,
        recompute=args.recompute_cache,
        use_cache=not args.no_cache,
        buff_size=args.buff_size,
        morphology_gas=args.morphology_gas,
    )
    angular_momentum = load_angular_momentum_data()

    build_figure(morphology, angular_momentum, output_prefix=output_prefix, palette=OKABE_ITO, save_pdf=True)
    if args.skip_muted_preview:
        print(f"Saved {output_prefix.with_suffix('.png')} and {output_prefix.with_suffix('.pdf')}")
    else:
        build_figure(
            morphology,
            angular_momentum,
            output_prefix=output_prefix.with_name(output_prefix.name + "_muted_preview"),
            palette=MUTED,
            save_pdf=False,
        )
        print(
            f"Saved {output_prefix.with_suffix('.png')}, {output_prefix.with_suffix('.pdf')}, "
            f"and {output_prefix.with_name(output_prefix.name + '_muted_preview').with_suffix('.png')}"
        )


if __name__ == "__main__":
    main()
