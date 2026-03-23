"""Visualization tests for dispersive GRIN ray drawing."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland.materials import GradientMaterial
from optiland.optic import Optic

matplotlib.use("Agg")

SPEED_OF_LIGHT_UM_THz = 299.792458


class DrudeGradientMaterial(GradientMaterial):
    """Test-only GRIN material with a Drude-like dispersive base index.

    The visualization stack traces geometric rays, so this helper combines:

    - a wavelength-dependent Drude-style baseline index, and
    - a quadratic radial GRIN term for continuous ray bending.

    This keeps the test self-contained while exercising `optic.draw()` with
    dispersive GRIN propagation.

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

        base_index = self._drude_index(wavelength)
        return base_index + self.nr2 * r_sq

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
        base_index = self._drude_index(wavelength)
        n = base_index + self.nr2 * r_sq

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
    """Builds an oblique, parallel Gaussian-apodized beam test optic."""
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


def _max_linear_deviation(line) -> float:
    """Returns the maximum deviation from the line joining both endpoints."""
    z_data = np.asarray(line.get_xdata(), dtype=float)
    x_data = np.asarray(line.get_ydata(), dtype=float)
    mask = np.isfinite(z_data) & np.isfinite(x_data)
    z_data = z_data[mask]
    x_data = x_data[mask]

    if z_data.size < 3 or np.isclose(z_data[-1], z_data[0]):
        return 0.0

    x_linear = x_data[0] + (x_data[-1] - x_data[0]) * (
        (z_data - z_data[0]) / (z_data[-1] - z_data[0])
    )
    return float(np.max(np.abs(x_data - x_linear)))


def test_draw_drude_grin_oblique_gaussian_beam(set_test_backend):
    """`draw()` should render dispersive GRIN paths for an angled Gaussian beam."""
    optic, wavelengths = _build_drude_grin_optic()

    fig, ax = optic.draw(
        wavelengths="all",
        num_rays=7,
        distribution="line_x",
        projection="XZ",
        title="Drude GRIN Gaussian beam draw test",
    )

    ray_lines = [line for line in ax.get_lines() if len(line.get_xdata()) > 2]

    assert len(ray_lines) >= len(wavelengths) * 7
    assert max(len(line.get_xdata()) for line in ray_lines) > 20
    assert len({line.get_color() for line in ray_lines}) >= len(wavelengths)
    assert any(_max_linear_deviation(line) > 1e-3 for line in ray_lines)
    assert ax.get_xlabel() == "Z [mm]"
    assert ax.get_ylabel() == "X [mm]"
    assert ax.get_title() == "Drude GRIN Gaussian beam draw test"

    plt.close(fig)
