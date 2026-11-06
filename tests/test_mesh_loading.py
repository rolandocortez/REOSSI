"""Mesh inputs obey metadata limits before large array allocation."""

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from reossi.simulation.geometry import load_mesh, make_surface


def test_npz_count_limit_precedes_numpy_loading(tmp_path, monkeypatch):
    mesh = make_surface(columns=3, rows=3)
    path = tmp_path / "mesh.npz"
    np.savez_compressed(path, vertices=mesh.vertices, faces=mesh.faces)
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **kw: pytest.fail("Count guard should reject before loading"),
    )
    with pytest.raises(ValueError, match="count limit"):
        load_mesh(path, max_faces=1)


def test_npz_declared_shape_cannot_bypass_uncompressed_guard(tmp_path):
    path = tmp_path / "declared.npz"
    stream = BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f8", "fortran_order": False, "shape": (10**9, 3)}
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("vertices.npy", stream.getvalue())
        archive.writestr("faces.npy", stream.getvalue())
    with pytest.raises(ValueError):
        load_mesh(path)


def test_standard_obj_mesh_is_loaded_as_geometry_only(tmp_path):
    pytest.importorskip("trimesh")
    path = tmp_path / "plane.obj"
    path.write_text("v -1 -1 2\nv 1 -1 2\nv 1 1 2\nv -1 1 2\nf 1 2 3\nf 1 3 4\n")
    mesh = load_mesh(path)
    assert mesh.vertices.shape == (4, 3) and mesh.faces.shape == (2, 3)


def test_point_only_ply_is_not_invented_as_a_surface(tmp_path):
    pytest.importorskip("trimesh")
    path = tmp_path / "points.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 1\n1 0 1\n0 1 1\n"
    )
    with pytest.raises(ValueError, match="point clouds"):
        load_mesh(path)
