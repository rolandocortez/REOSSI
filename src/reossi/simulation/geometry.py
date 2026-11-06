"""Small triangular scenes and calibrated pinhole devices for digital scans."""

from dataclasses import dataclass, field
from math import isfinite, prod
from numbers import Integral
from pathlib import Path
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    """A surface in world units, with vertices ``(V,3)`` and faces ``(F,3)``.

    Faces are rendered from either side. Degeneracy uses the angle between
    normalized edges, so changing length units does not reject small valid
    triangles. A point cloud without faces is not a surface and is rejected;
    its reconstruction is a separate modeling choice.
    """

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("vertices must be a nonempty (V, 3) array")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("faces must be a nonempty (F, 3) triangle array")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("Face indices must be integers")
        if (
            not np.isfinite(vertices).all()
            or faces.min() < 0
            or faces.max() >= len(vertices)
        ):
            raise ValueError("Mesh contains nonfinite vertices or invalid face indices")
        triangles = vertices[faces]
        with np.errstate(over="ignore", invalid="ignore"):
            first_edges = triangles[:, 1] - triangles[:, 0]
            second_edges = triangles[:, 2] - triangles[:, 0]
            first_lengths = np.hypot.reduce(first_edges, axis=1)
            second_lengths = np.hypot.reduce(second_edges, axis=1)
        if (
            not np.isfinite(first_lengths).all()
            or not np.isfinite(second_lengths).all()
            or np.any(first_lengths == 0)
            or np.any(second_lengths == 0)
        ):
            raise ValueError(
                "Mesh contains degenerate or unrepresentable triangle edges"
            )
        relative_areas = np.linalg.norm(
            np.cross(
                first_edges / first_lengths[:, None],
                second_edges / second_lengths[:, None],
            ),
            axis=1,
        )
        if np.any(relative_areas <= np.finfo(float).eps):
            raise ValueError("Mesh contains degenerate triangles")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces.astype(np.int64, copy=False))

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]


def load_mesh(
    path: str | Path,
    *,
    max_vertices: int = 500_000,
    max_faces: int = 200_000,
    max_file_mib: float = 64,
) -> TriangleMesh:
    """Load NPZ vertices/faces, or a mesh file through optional ``trimesh``.

    NPZ needs no additional dependency. OBJ, STL and PLY support is provided
    by ``pip install 'reossi[mesh]'``. Point-only files are rejected.
    """
    path = Path(path)
    if (
        any(
            not isinstance(v, Integral) or isinstance(v, bool) or v < 1
            for v in (max_vertices, max_faces)
        )
        or not isfinite(max_file_mib)
        or max_file_mib <= 0
    ):
        raise ValueError("Mesh limits must be finite and positive")
    byte_limit = max_file_mib * 1024**2
    if path.stat().st_size > byte_limit:
        raise ValueError(f"Mesh file exceeds max_file_mib={max_file_mib:g}")
    if path.suffix.lower() == ".npz":
        with ZipFile(path) as archive:
            total = 0
            for name, maximum in (("vertices", max_vertices), ("faces", max_faces)):
                entry = archive.getinfo(name + ".npy")
                total += entry.file_size
                if total > byte_limit:
                    raise ValueError("Mesh arrays exceed the uncompressed memory limit")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError("Mesh NPZ requires NPY format version 1 or 2")
                if len(shape) != 2 or shape[1] != 3 or not 0 < shape[0] <= maximum:
                    raise ValueError(f"{name} exceeds its shape/count limit")
                if dtype.hasobject or prod(shape) * dtype.itemsize > byte_limit:
                    raise ValueError("Mesh array dtype or declared size is unsupported")
        with np.load(path, allow_pickle=False) as data:
            return TriangleMesh(data["vertices"], data["faces"])
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "Install reossi[mesh] to load OBJ, STL or PLY meshes"
        ) from error
    if path.suffix.lower() not in {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}:
        raise ValueError("Use an NPZ, OBJ, STL, PLY, OFF, GLB or GLTF mesh file")
    mesh = trimesh.load_mesh(path, process=False, skip_materials=True)
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or not len(mesh.vertices)
        or not len(mesh.faces)
    ):
        raise ValueError(
            "Input must contain one triangle mesh; point clouds need a surface"
        )
    if len(mesh.vertices) > max_vertices or len(mesh.faces) > max_faces:
        raise ValueError("Loaded mesh exceeds the vertex/face budget")
    return TriangleMesh(mesh.vertices, mesh.faces)


def make_surface(
    *,
    width: float = 1.6,
    height: float = 1.2,
    depth: float = 2.0,
    columns: int = 2,
    rows: int = 2,
    tilt: float = 0.0,
    curvature: float = 0.0,
) -> TriangleMesh:
    """Build a plane or tessellated height field ``z=depth+tilt*x+curvature*x²``.

    Width/height/depth share world units. Curvature is inverse world units;
    this is known geometry, not an optical distortion applied to identifiers.
    """
    if min(width, height, depth) <= 0 or any(
        not isinstance(v, Integral) or isinstance(v, bool) or v < 2
        for v in (columns, rows)
    ):
        raise ValueError("Surface dimensions must be positive; rows/columns >= 2")
    if not np.isfinite([width, height, depth, tilt, curvature]).all():
        raise ValueError("Surface parameters must be finite")
    x, y = np.meshgrid(
        np.linspace(-width / 2, width / 2, columns),
        np.linspace(-height / 2, height / 2, rows),
    )
    vertices = np.stack([x, y, depth + tilt * x + curvature * x**2], axis=-1).reshape(
        -1, 3
    )
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            index = row * columns + column
            faces.extend(
                [
                    [index, index + 1, index + columns],
                    [index + 1, index + columns + 1, index + columns],
                ]
            )
    return TriangleMesh(vertices, np.array(faces))


@dataclass(frozen=True)
class PinholeDevice:
    """Calibrated camera/projector using OpenCV axes and world-to-device pose.

    Integer image coordinates are pixel centers. Rotation/translation satisfy
    ``X_device = rotation @ X_world + translation``. No lens distortion is
    modeled by the initial renderer; physical calibration can include it.
    """

    image_size: tuple[int, int]
    matrix: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float).reshape(-1)
        if len(self.image_size) != 2 or any(
            not isinstance(v, int) or v < 1 for v in self.image_size
        ):
            raise ValueError("image_size must contain two positive integers")
        if (
            matrix.shape != (3, 3)
            or rotation.shape != (3, 3)
            or translation.shape != (3,)
        ):
            raise ValueError(
                "Expected 3x3 intrinsics/rotation and a 3-vector translation"
            )
        if not all(np.isfinite(v).all() for v in (matrix, rotation, translation)):
            raise ValueError("Device calibration must be finite")
        if (
            matrix[0, 0] <= 0
            or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
        ):
            raise ValueError(
                "Intrinsics require positive focal lengths and last row [0,0,1]"
            )
        if np.linalg.matrix_rank(matrix) < 3:
            raise ValueError("Intrinsics must be nonsingular")
        if not np.allclose(
            rotation @ rotation.T, np.eye(3), atol=1e-8
        ) or not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("rotation must be a proper orthonormal matrix")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def center(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return origins and unit directions in world coordinates, ``(H*W,3)``."""
        width, height = self.image_size
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        image_points = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
        directions = image_points @ np.linalg.inv(self.matrix).T @ self.rotation
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return np.broadcast_to(self.center, directions.shape), directions

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return continuous device pixels and device-frame Z for world points."""
        device_points = np.asarray(points) @ self.rotation.T + self.translation
        homogeneous = device_points @ self.matrix.T
        with np.errstate(invalid="ignore", divide="ignore"):
            pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
        return pixels, device_points[:, 2]


def make_device(
    image_size: tuple[int, int],
    *,
    focal_length: float | None = None,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PinholeDevice:
    """Create an axis-aligned pinhole device with centered principal point."""
    width, height = image_size
    focal = float(focal_length if focal_length is not None else width)
    matrix = np.array(
        [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]
    )
    return PinholeDevice(image_size, matrix, translation=-np.array(center))


def relative_calibration(camera: PinholeDevice, projector: PinholeDevice):
    """Return the camera-to-projector R/T used by OpenCV stereo calibration."""
    rotation = projector.rotation @ camera.rotation.T
    translation = projector.translation - rotation @ camera.translation
    return rotation, translation.reshape(3, 1)


def look_at_rotation(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orient a device toward a target, using world +Y as image-down direction."""
    center, target = np.asarray(center, dtype=float), np.asarray(target, dtype=float)
    if (
        center.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Look-at center and target must be finite three-vectors")
    forward = target - center
    if np.linalg.norm(forward) < 1e-12:
        raise ValueError("Look-at center and target must differ")
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 1.0, 0.0], forward)
    if np.linalg.norm(right) < 1e-10:
        raise ValueError("View direction must not be parallel to world Y")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])
