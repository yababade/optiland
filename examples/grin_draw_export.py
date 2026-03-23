"""Build a simple GRIN ray-tracing case with ``draw()`` and export an image."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import optiland.backend as be
from optiland.materials import GradientMaterial
from optiland.optic import Optic

matplotlib.use("Agg")


def build_simple_grin_optic() -> Optic:
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


def export_grin_draw(output_path: str | Path, show: bool = False) -> Path:
    """Renders the GRIN case with ``draw()`` and exports it to disk.

    Args:
        output_path: The file path for the exported image.
        show: If ``True``, display the figure after saving.

    Returns:
        The resolved output path.
    """
    optic = build_simple_grin_optic()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = optic.draw(
        wavelengths="primary",
        num_rays=5,
        distribution="line_x",
        projection="XZ",
        title="Simple GRIN ray trace",
    )
    fig.canvas.draw()
    fig.savefig(output, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render a simple GRIN draw() example and export an image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="grin_draw_export.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    args = parser.parse_args()

    output = export_grin_draw(args.output, show=args.show)
    print(f"Saved GRIN draw image to: {output}")


if __name__ == "__main__":
    main()
