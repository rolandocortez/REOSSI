"""Bounded, direct-only ray queries for small triangular research scenes.

The intersection calculation uses the Möller–Trumbore barycentric equations.
It intentionally uses no acceleration structure: memory stays proportional
to the ray count, and an explicit ray/triangle test budget bounds the work.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np

from .geometry import PinholeDevice, TriangleMesh


@dataclass(frozen=True)
class RayHits:
    """Nearest intersections; missed faces=-1, distances=inf, points=NaN."""

    distances: np.ndarray
    face_indices: np.ndarray
    points: np.ndarray


def intersect_mesh(
    mesh: TriangleMesh,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    minimum_distance: float = 1e-8,
    epsilon: float = 1e-10,
    max_triangle_tests: int = 20_000_000,
) -> RayHits:
    """Find the nearest positive two-sided triangle hit for each unit ray.

    Inputs have shape ``(N,3)``; a common origin may be a broadcast array.
    Faces are visited in input order, preserving the first hit on shared
    edges when equal distances occur. ``epsilon`` is dimensionless: it bounds
    the determinant relative to the product of edge lengths and supplies the
    barycentric slack. ``minimum_distance`` remains an explicit world-unit
    distance. The O(N*F) budget is checked before allocating ray intermediates.
    Large meshes need a different renderer.
    """
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("Ray origins/directions must have matching (N,3) shapes")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all():
        raise ValueError("Rays must be finite")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-8):
        raise ValueError("Ray directions must be unit vectors")
    if (
        not isfinite(minimum_distance)
        or not isfinite(epsilon)
        or minimum_distance < 0
        or epsilon <= 0
        or not isinstance(max_triangle_tests, Integral)
        or isinstance(max_triangle_tests, bool)
        or max_triangle_tests < 1
    ):
        raise ValueError("Invalid ray tolerance or triangle-test budget")
    if len(directions) * len(mesh.faces) > max_triangle_tests:
        raise ValueError(
            "Ray/triangle budget exceeded; reduce scene resolution or mesh size"
        )
    distances = np.full(len(directions), np.inf)
    face_indices = np.full(len(directions), -1, dtype=np.int64)
    for face_index, triangle in enumerate(mesh.triangles):
        vertex, second, third = triangle
        first_edge, second_edge = second - vertex, third - vertex
        with np.errstate(over="ignore", under="ignore"):
            edge_scale = np.hypot.reduce(first_edge) * np.hypot.reduce(second_edge)
        if not np.isfinite(edge_scale) or edge_scale <= 0:
            raise ValueError(
                "Triangle scale exceeds ray arithmetic; rescale world units"
            )
        cross = np.cross(directions, second_edge)
        determinant = cross @ first_edge
        possible = np.abs(determinant) > epsilon * edge_scale
        inverse = np.divide(
            1.0, determinant, out=np.zeros_like(determinant), where=possible
        )
        offset = origins - vertex
        u = np.einsum("ij,ij->i", offset, cross) * inverse
        other_cross = np.cross(offset, first_edge)
        v = np.einsum("ij,ij->i", directions, other_cross) * inverse
        distance = (other_cross @ second_edge) * inverse
        hits = (
            possible
            & (u >= -epsilon)
            & (v >= -epsilon)
            & (u + v <= 1 + epsilon)
            & (distance > minimum_distance)
            & (distance < distances)
        )
        distances[hits] = distance[hits]
        face_indices[hits] = face_index
    points = np.full(directions.shape, np.nan)
    hit_mask = face_indices >= 0
    points[hit_mask] = (
        origins[hit_mask] + directions[hit_mask] * distances[hit_mask, None]
    )
    return RayHits(distances, face_indices, points)


@dataclass(frozen=True)
class SceneCorrespondences:
    """Camera-visible geometry and projector visibility, in camera image order.

    ``points_world``/``points_camera`` have shape ``(H,W,3)``. Projector pixels
    have shape ``(H,W,2)`` and remain continuous; ``illuminated`` excludes
    projector occlusion, out-of-frame projections and points behind it.
    """

    points_world: np.ndarray
    points_camera: np.ndarray
    projector_pixels: np.ndarray
    camera_visible: np.ndarray
    illuminated: np.ndarray
    face_indices: np.ndarray


def scene_correspondences(
    mesh: TriangleMesh,
    camera: PinholeDevice,
    projector: PinholeDevice,
    *,
    max_triangle_tests: int = 20_000_000,
    visibility_tolerance: float = 1e-6,
) -> SceneCorrespondences:
    """Trace camera rays, then test whether the projector sees each surface.

    Camera visibility and projector visibility are distinct masks. Shadow
    queries start at the projector center; their nearest surface must coincide
    with the camera-visible point. ``visibility_tolerance`` is dimensionless:
    it is the relative distance tolerance and multiplies the scene extent for
    the absolute tolerance. Minimum ray distance is likewise scaled by that
    extent. Extent is the largest vertex distance from the first mesh vertex,
    invariant to a rigid world transform and proportional to length units.
    This does not simulate interreflection, transparency or subsurface light
    transport. Exact viewport-boundary ties retain floating-point sensitivity.
    """
    width, height = camera.image_size
    if (
        not isinstance(max_triangle_tests, Integral)
        or isinstance(max_triangle_tests, bool)
        or max_triangle_tests < 1
    ):
        raise ValueError("max_triangle_tests must be a positive integer")
    if 2 * width * height * len(mesh.faces) > max_triangle_tests:
        raise ValueError(
            "Combined camera/shadow ray budget exceeded; use a smaller scene"
        )
    if not isfinite(visibility_tolerance) or visibility_tolerance <= 0:
        raise ValueError("visibility_tolerance must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        extent = float(
            np.max(np.hypot.reduce(mesh.vertices - mesh.vertices[0], axis=1))
        )
    if not isfinite(extent) or extent <= 0:
        raise ValueError("Scene extent must be finite and positive")
    minimum_distance = 1e-8 * extent
    absolute_visibility_tolerance = visibility_tolerance * extent
    origins, rays = camera.rays()
    hits = intersect_mesh(
        mesh,
        origins,
        rays,
        minimum_distance=minimum_distance,
        max_triangle_tests=max_triangle_tests,
    )
    visible = hits.face_indices >= 0
    projector_pixels = np.full((len(rays), 2), np.nan)
    illuminated = np.zeros(len(rays), dtype=bool)
    visible_indices = np.flatnonzero(visible)
    if len(visible_indices):
        points = hits.points[visible_indices]
        pixels, depth = projector.project(points)
        projector_pixels[visible_indices] = pixels
        projector_width, projector_height = projector.image_size
        in_frame = (
            (depth > 0)
            & (pixels[:, 0] >= -0.5)
            & (pixels[:, 1] >= -0.5)
            & (pixels[:, 0] < projector_width - 0.5)
            & (pixels[:, 1] < projector_height - 0.5)
        )
        candidates = visible_indices[in_frame]
        if len(candidates):
            offsets = hits.points[candidates] - projector.center
            lengths = np.linalg.norm(offsets, axis=1)
            nonzero = lengths > minimum_distance
            candidates, offsets, lengths = (
                candidates[nonzero],
                offsets[nonzero],
                lengths[nonzero],
            )
            shadow_hits = intersect_mesh(
                mesh,
                np.broadcast_to(projector.center, offsets.shape),
                offsets / lengths[:, None],
                minimum_distance=minimum_distance,
                max_triangle_tests=max_triangle_tests,
            )
            illuminated[candidates] = np.isclose(
                shadow_hits.distances,
                lengths,
                rtol=visibility_tolerance,
                atol=absolute_visibility_tolerance,
            )
    camera_points = hits.points @ camera.rotation.T + camera.translation
    return SceneCorrespondences(
        hits.points.reshape(height, width, 3),
        camera_points.reshape(height, width, 3),
        projector_pixels.reshape(height, width, 2),
        visible.reshape(height, width),
        illuminated.reshape(height, width),
        hits.face_indices.reshape(height, width),
    )
