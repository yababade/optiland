"""Simple GRIN ray-tracing visualization test using ``draw()``."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.materials import GradientMaterial
from optiland.optic import Optic

matplotlib.use("Agg")


def _build_simple_grin_optic() -> Optic:
    """Builds a simple optic containing a single GRIN slab."""
    optic = Optic()
    optic.add_surface(index=0, radius=be.inf, thickness=be.inf, comment="Object")
    optic.add_surface(
        index=1,
        radius=be.inf,
        thickness=8.0,
        material=GradientMaterial(
            n0=1.62,
            nr2=-0.025,
            step_size=0.002,
        ),
        is_stop=True,
        comment="GRIN entrance plane",
    )
    optic.add_surface(
        index=2,
        radius=be.inf,
        thickness=20.0,
        material="air",
        comment="GRIN exit plane",
    )
    optic.add_surface(index=3, radius=be.inf, comment="Image")

    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.set_apodization("GaussianApodization", sigma=0.4)
    optic.set_field_type(field_type="angle")
    optic.add_field(x=5.0, y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


def test_draw_simple_grin_raytrace_saves_image(set_test_backend, tmp_path):
    """`draw()` should render and save a simple GRIN ray-tracing image."""
    optic = _build_simple_grin_optic()

    fig, ax = optic.draw(
        wavelengths="primary",
        num_rays=5,
        distribution="line_x",
        projection="XZ",
        title="Simple GRIN ray trace",
    )

    output_path = tmp_path / "simple_grin_raytrace.png"
    fig.canvas.draw()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")

    ray_lines = [line for line in ax.get_lines() if len(line.get_xdata()) > 2]

    assert len(ray_lines) >= 5
    assert max(len(line.get_xdata()) for line in ray_lines) > 10
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert ax.get_xlabel() == "Z [mm]"
    assert ax.get_ylabel() == "X [mm]"
    assert ax.get_title() == "Simple GRIN ray trace"

    plt.close(fig)
