"""WebAgg-backed 3D visualization test for dispersive GRIN ray tracing."""
from __future__ import annotations

from io import BytesIO

import matplotlib
import numpy as np
from matplotlib.backends.backend_webagg_core import (
    FigureCanvasWebAggCore as FigureCanvasWebAgg,
)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import optiland.backend as be
from optiland.materials import GradientMaterial
from optiland.optic import Optic

matplotlib.use("Agg")

SPEED_OF_LIGHT_UM_THz = 299.792458


class DrudeGradientMaterial(GradientMaterial):
    """Test-only GRIN material with a Drude-like dispersive base index.

    Args:
        epsilon_inf: High-frequency dielectric constant.
        plasma_frequency: Effective plasma frequency in THz.
        collision_frequency: Effective damping frequency in THz.
        nr2: Radial GRIN coefficient.
        step_size: RK4 propagation step size in millimeters.
    """

    def __init__(
        self,
        epsilon_inf: float,
        plasma_frequency: float,
        collision_frequency: float,
        nr2: float,
        step_size: float = 0.002,
    ) -> None:
        super().__init__(n0=0.0, nr2=nr2, step_size=step_size)
        self.epsilon_inf = epsilon_inf
        self.plasma_frequency = plasma_frequency
        self.collision_frequency = collision_frequency

    def _calculate_n(
        self, wavelength: float | be.ndarray, **kwargs
    ) -> float | be.ndarray:
        """Calculates the Drude-GRIN refractive index."""
        x = kwargs.get("x", 0.0)
        y = kwargs.get("y", 0.0)
        r_sq = x**2 + y**2
        return self._drude_index(wavelength) + self.nr2 * r_sq

    def get_index_and_gradient(
        self,
        x: float | be.ndarray,
        y: float | be.ndarray,
        z: float | be.ndarray,
        wavelength: float | be.ndarray,
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray, be.ndarray]:
        """Calculates the local index and spatial gradient."""
        x = be.atleast_1d(be.array(x))
        y = be.atleast_1d(be.array(y))
        _ = z

        r_sq = x**2 + y**2
        n = self._drude_index(wavelength) + self.nr2 * r_sq
        dn_dx = 2.0 * self.nr2 * x
        dn_dy = 2.0 * self.nr2 * y
        dn_dz = be.zeros_like(dn_dx)
        return n, dn_dx, dn_dy, dn_dz

    def _drude_index(self, wavelength: float | be.ndarray) -> float | be.ndarray:
        """Converts a Drude-like permittivity into a real refractive index."""
        wavelength_array = be.array(wavelength)
        frequency = SPEED_OF_LIGHT_UM_THz / wavelength_array
        permittivity = self.epsilon_inf - (
            self.plasma_frequency**2
            / (frequency**2 + self.collision_frequency**2)
        )
        return be.sqrt(be.clip(permittivity, 1.05, None))


def _sample_gaussian_wavelengths(
    seed: int = 7,
    sample_size: int = 3,
    mean_frequency: float = 540.0,
    sigma_frequency: float = 18.0,
) -> list[float]:
    """Samples wavelengths by drawing frequencies from a Gaussian law."""
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(
        loc=mean_frequency,
        scale=sigma_frequency,
        size=sample_size,
    )
    frequencies = np.clip(frequencies, 450.0, None)
    wavelengths = SPEED_OF_LIGHT_UM_THz / frequencies
    return sorted(float(value) for value in wavelengths)


def _build_drude_grin_optic() -> tuple[Optic, list[float]]:
    """Builds the dispersive GRIN optic for the WebAgg 3D test."""
    wavelengths = _sample_gaussian_wavelengths()
    material = DrudeGradientMaterial(
        epsilon_inf=4.4,
        plasma_frequency=620.0,
        collision_frequency=85.0,
        nr2=-0.03,
    )

    optic = Optic()
    optic.add_surface(index=0, radius=be.inf, thickness=be.inf, comment="Object")
    optic.add_surface(
        index=1,
        radius=be.inf,
        thickness=6.0,
        material=material,
        is_stop=True,
        comment="Drude GRIN entrance plane",
    )
    optic.add_surface(
        index=2,
        radius=be.inf,
        thickness=25.0,
        material="air",
        comment="Drude GRIN exit plane",
    )
    optic.add_surface(index=3, radius=be.inf, comment="Image")

    optic.set_aperture(aperture_type="EPD", value=2.5)
    optic.set_apodization("GaussianApodization", sigma=0.35)
    optic.set_field_type(field_type="angle")
    optic.add_field(x=8.0, y=0.0)

    for index, wavelength in enumerate(wavelengths):
        optic.add_wavelength(value=wavelength, is_primary=(index == 1))

    return optic, wavelengths


def _max_linear_deviation(
    z_data: np.ndarray,
    x_data: np.ndarray,
) -> float:
    """Returns the maximum deviation from the line joining both endpoints."""
    mask = np.isfinite(z_data) & np.isfinite(x_data)
    z_data = z_data[mask]
    x_data = x_data[mask]

    if z_data.size < 3 or np.isclose(z_data[-1], z_data[0]):
        return 0.0

    x_linear = x_data[0] + (x_data[-1] - x_data[0]) * (
        (z_data - z_data[0]) / (z_data[-1] - z_data[0])
    )
    return float(np.max(np.abs(x_data - x_linear)))


def _render_webagg_3d_image(
    optic: Optic,
    wavelengths: list[float],
    output_path,
) -> tuple[Figure, np.ndarray, int]:
    """Renders traced GRIN rays onto a WebAgg-backed 3D figure."""
    fig = Figure(figsize=(8, 6))
    canvas = FigureCanvasWebAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    total_paths = 0
    color_map = matplotlib.colormaps["plasma"]

    for index, wavelength in enumerate(wavelengths):
        rays = optic.trace(
            8.0,
            0.0,
            wavelength,
            num_rays=7,
            distribution="line_x",
            record_path=True,
        )

        assert rays.has_paths()
        total_paths += len(rays.get_paths())
        color = color_map(index / max(len(wavelengths) - 1, 1))

        for ray_index, (x_path, y_path, z_path) in enumerate(rays.get_paths()):
            if z_path.size < 3:
                continue

            alpha = 1.0
            if rays.path_i and rays.path_i[ray_index]:
                alpha = float(np.clip(np.mean(rays.path_i[ray_index]), 0.2, 1.0))

            ax.plot(
                z_path,
                x_path,
                y_path,
                color=color,
                alpha=alpha,
                linewidth=1.2,
            )

    ax.set_xlabel("Z [mm]")
    ax.set_ylabel("X [mm]")
    ax.set_zlabel("Y [mm]")
    ax.set_title("WebAgg 3D Drude GRIN Gaussian beam")
    ax.view_init(elev=25.0, azim=-120.0)

    canvas.draw()
    rendered = np.asarray(canvas.buffer_rgba())

    png_buffer = BytesIO()
    canvas.print_png(png_buffer)
    output_path.write_bytes(png_buffer.getvalue())

    return fig, rendered, total_paths


def test_webagg_forwards_drude_grin_3d_image(set_test_backend, tmp_path):
    """WebAgg should forward a rendered 3D GRIN beam image."""
    optic, wavelengths = _build_drude_grin_optic()
    output_path = tmp_path / "webagg_drude_grin_gaussian_beam_3d.png"

    fig, rendered, total_paths = _render_webagg_3d_image(
        optic,
        wavelengths,
        output_path,
    )

    assert total_paths >= len(wavelengths) * 7
    assert rendered.ndim == 3
    assert rendered.shape[-1] == 4
    assert np.unique(rendered.reshape(-1, rendered.shape[-1]), axis=0).shape[0] > 1
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    for wavelength in wavelengths:
        rays = optic.trace(
            8.0,
            0.0,
            wavelength,
            num_rays=7,
            distribution="line_x",
            record_path=True,
        )
        assert any(
            _max_linear_deviation(z_path, x_path) > 1e-3
            for x_path, _, z_path in rays.get_paths()
            if z_path.size >= 3
        )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Z [mm]"
    assert ax.get_ylabel() == "X [mm]"
    assert ax.get_zlabel() == "Y [mm]"
    assert ax.get_title() == "WebAgg 3D Drude GRIN Gaussian beam"
