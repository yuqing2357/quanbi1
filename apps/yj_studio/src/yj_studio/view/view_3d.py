from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from .qt_vtk_view import QtVTKView


class View3D(QtVTKView):
    """Main 3D interpretation viewport."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self.set_background("#202124")
        self.add_axes()

    def reset_to_volume(self, shape: tuple[int, int, int]) -> None:
        nx, ny, nz = shape
        self.camera_position = [
            (float(nx) * 1.4, -float(ny) * 1.6, float(nz) * 1.2),
            (float(nx) / 2.0, float(ny) / 2.0, float(nz) / 2.0),
            (0.0, 0.0, 1.0),
        ]
        self.reset_camera()

