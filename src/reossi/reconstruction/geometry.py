# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND MIT
# Copyright 2026 Rolando Cortez Garcia
# Copyright (c) 2023 Guillaume Charron
# Original additions and inherited portions: see docs/licensing.md.
"""Camera-projector geometry adapted from Charron's MIT-licensed scanner.

The calculations, point order, clipping and in-place intrinsic scaling follow
the pinned reference. ``to_camera_frame`` comes from the local REOSSI scanner.
"""

import cv2
import numpy as np


class StereoTriangulator:
    """Triangulate decoded projector coordinates with the reference geometry.

    Image sizes are ``(width, height)``. Calibration translation determines the
    units of returned points. The supplied projector matrix is scaled in place,
    preserving the original constructor's behavior.
    """

    def __init__(
        self,
        camera_size=(1920, 1080),
        camera_matrix=None,
        camera_distortion=None,
        projector_size=(1920, 1080),
        projector_calibration_size=(1920, 1080),
        projector_matrix=None,
        projector_distortion=None,
        rotation=None,
        translation=None,
    ):
        self.camera_width, self.camera_height = camera_size
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion
        self.projector_width, self.projector_height = projector_size
        self.projector_matrix = projector_matrix
        self.projector_distortion = projector_distortion

        if projector_matrix is not None:
            calibration_width, calibration_height = projector_calibration_size
            scale_x = self.projector_width / calibration_width
            scale_y = self.projector_height / calibration_height
            self.projector_matrix[0, :] = self.projector_matrix[0, :] * scale_x
            self.projector_matrix[1, :] = self.projector_matrix[1, :] * scale_y

        self.translation = translation
        self.rotation = rotation

    def collect_correspondences(
        self, column_ids, row_ids, white_image, *, reject_negative=True
    ):
        """Return camera/projector points and colors in the historical order.

        Identifier maps have camera shape ``(height, width)``. ``white_image``
        supplies the corresponding three-channel colors, retained in its input
        channel order and converted to float64 in the range 0..1. Outputs have
        shapes ``(N, 2)``, ``(N, 2)`` and ``(N, 3)`` for nonempty input.

        The local scanner rejects identifiers <= -1. Set ``reject_negative``
        to False for the upstream rule, which rejects only the sentinel -1.
        Values above the projector range are clipped as in the reference.
        """
        camera_points = []
        projector_points = []
        colors = []

        for column in range(self.camera_width):
            for row in range(self.camera_height):
                column_id = column_ids[row, column]
                row_id = row_ids[row, column]
                if reject_negative:
                    invalid = column_id <= -1 or row_id <= -1
                else:
                    invalid = column_id == -1 or row_id == -1
                if invalid:
                    continue
                camera_points.append([column, row])
                column_id = min(self.projector_width - 1, column_id)
                row_id = min(self.projector_height - 1, row_id)
                projector_points.append([column_id, row_id])
                colors.append(white_image[row, column, :])

        camera_points = np.array(camera_points, dtype=np.float32)
        projector_points = np.array(projector_points, dtype=np.float32)
        colors = np.array(colors).astype(np.float64) / 255.0
        return camera_points, projector_points, colors

    def triangulate(self, camera_points, projector_points):
        """Return a ``(3, N)`` point array using the reference sine rule.

        Input coordinates have shape ``(N, 2)``. The returned vectors originate
        at the camera center and use axes rotated into projector orientation.
        No additional checks or filtering are applied to degenerate rays.
        """
        camera_rays = cv2.convertPointsToHomogeneous(
            cv2.undistortPoints(
                np.expand_dims(camera_points, axis=1),
                self.camera_matrix,
                self.camera_distortion,
                R=self.rotation,
            )
        )[:, 0].T
        projector_rays = cv2.convertPointsToHomogeneous(
            cv2.undistortPoints(
                np.expand_dims(projector_points, axis=1),
                self.projector_matrix,
                self.projector_distortion,
            )
        )[:, 0].T
        translation = self.translation[:, 0]

        # The reference attributes this calculation to caoandong/Projector_Calibration.
        baseline = np.linalg.norm(translation)
        normalized_rays = camera_rays / np.linalg.norm(camera_rays, axis=0)
        alpha = np.arccos(np.dot(-translation, normalized_rays) / baseline)
        beta = np.arccos(
            np.dot(translation, projector_rays)
            / (baseline * np.linalg.norm(projector_rays, axis=0))
        )
        gamma = np.pi - alpha - beta
        distances = baseline * np.sin(beta) / np.sin(gamma)
        return normalized_rays * distances

    def to_camera_frame(self, points):
        """Rotate ``(3, N)`` point vectors into the original camera axes.

        This is the local scanner's inverse-rotation helper; it does not add a
        translation or change the camera-centered origin.
        """
        inverse_rotation = np.linalg.inv(self.rotation)
        return inverse_rotation @ points

    def filter_points(self, points, colors, threshold=0.5):
        """Retain points strictly inside the reference axis-aligned cube.

        ``points`` has shape ``(3, N)`` and ``colors`` has shape ``(N, 3)``.
        The threshold uses the calibration translation's length unit.
        """
        mask = (
            (points[2] < threshold)
            & (points[2] > -threshold)
            & (points[1] < threshold)
            & (points[1] > -threshold)
            & (points[0] < threshold)
            & (points[0] > -threshold)
        )
        return points[:, mask], colors[mask]
