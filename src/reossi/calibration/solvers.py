# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 AND MIT
# Copyright 2026 Rolando Cortez Garcia
# Copyright (c) 2023 Guillaume Charron
# Original additions and inherited portions: see docs/licensing.md.
"""Calibration adapted from Charron's MIT-licensed reference scanner.

The reference board definitions, OpenCV calls, initial matrices, criteria,
image discovery order and return values are retained. This module does not
open a camera or run calibration when imported.
"""

import glob
import os
from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np


def _valid_homography(matrix) -> bool:
    """Check the finite floating 3-by-3 contract used by OpenCV transforms."""
    return (
        isinstance(matrix, np.ndarray)
        and matrix.shape == (3, 3)
        and matrix.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        and bool(np.isfinite(matrix).all())
    )


@dataclass(frozen=True)
class ChArUcoBoardConfig:
    """Describe a physical board; lengths determine calibration length units.

    Defaults match Charron's board. ``local_stereo`` reproduces the local
    stereo detector's board definition only. The historical local camera
    calibrator detected with 0.045/0.035 lengths but calibrated with
    0.025/0.020, so that preset does not establish local-camera equivalence.
    """

    squares: tuple[int, int] = (5, 7)
    square_length: float = 0.04
    marker_length: float = 0.02
    dictionary_id: int = cv2.aruco.DICT_6X6_250

    def __post_init__(self):
        if (
            not isinstance(self.squares, (tuple, list))
            or len(self.squares) != 2
            or any(not isinstance(value, int) or value < 2 for value in self.squares)
        ):
            raise ValueError("A ChArUco board needs two integer dimensions >= 2")
        if not (
            np.isfinite(self.square_length)
            and np.isfinite(self.marker_length)
            and 0 < self.marker_length < self.square_length
        ):
            raise ValueError("Board lengths must satisfy 0 < marker < square")
        dictionaries = {
            getattr(cv2.aruco, name)
            for name in dir(cv2.aruco)
            if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
        }
        if (
            not isinstance(self.dictionary_id, (int, np.integer))
            or isinstance(self.dictionary_id, bool)
            or self.dictionary_id not in dictionaries
        ):
            raise ValueError(
                "dictionary_id must select an OpenCV predefined dictionary"
            )

    @classmethod
    def local_stereo(cls) -> "ChArUcoBoardConfig":
        """Return the local stereo detector's 10-by-7 board definition."""
        return cls((10, 7), 0.045, 0.035, cv2.aruco.DICT_4X4_100)


class CameraCalibrator:
    """Calibrate the reference 5-by-7 ChArUco board with 0.04/0.02 lengths.

    Input sizes are in pixels; board lengths define pose translation units.
    Constructor parameters and method contracts follow the pinned reference.
    """

    def __init__(
        self,
        cam_width=1920,
        cam_height=1080,
        cam_mtx=None,
        cam_dist=None,
        *,
        board: ChArUcoBoardConfig = ChArUcoBoardConfig(),
        image_pattern: str = "*.jpg",
    ):

        # Define the charuco board parameters
        self.board_config = board
        self.image_pattern = image_pattern
        self.dictionary = cv2.aruco.getPredefinedDictionary(board.dictionary_id)
        self.charuco_board = cv2.aruco.CharucoBoard(
            board.squares, board.square_length, board.marker_length, self.dictionary
        )
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        # Camera parameters (image size and intrinsic parameters)
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

    def detect_markers(self, frame, gray, draw=True):
        """
        Detect markers and corners of the charuco board in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected markers
        gray : Mat
            gray version of the image
        draw : bool, optional (default = True)
            draw detected markers or not

        Returns:
        -------
        ret : bool
            True if at least one marker and the board were detected
        frame : Mat
            Image with detected markers
        allCorners : list
            Corners of the detected markers
        allIds : list
            Ids of the detected markers
        """
        rvec = None
        tvec = None
        ret = False
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # Detect markers and corners
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )
        allCorners = []
        allIds = []
        if corners is not None and len(corners) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3, 3), (-1, -1), criteria)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board
            )
            if charuco_ids is not None and len(charuco_ids) > 0:
                if draw:
                    frame = cv2.aruco.drawDetectedCornersCharuco(
                        frame, charuco_corners, charuco_ids
                    )
                if self.cam_mtx is not None and self.cam_dist is not None:
                    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners,
                        charuco_ids,
                        self.charuco_board,
                        self.cam_mtx,
                        self.cam_dist,
                        None,
                        None,
                    )

                    if ret and draw:
                        frame = cv2.drawFrameAxes(
                            frame, self.cam_mtx, self.cam_dist, rvec, tvec, 0.1
                        )

                allCorners = charuco_corners
                allIds = charuco_ids

        return ret, frame, allCorners, allIds

    def calibrate(
        self,
        calibration_folder,
        calib_flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        *,
        minimum_views=0,
        criteria=None,
        verbose=True,
    ):
        """
        Calibrate the camera using the charuco board

        Parameters:
        ----------
        calibration_folder : str
            path to the folder containing the calibration images
        calib_flags : int, optional (default = cv2.CALIB_USE_INTRINSIC_GUESS)
            flags for the calibration function

        Returns:
        -------
        cam_mtx : Mat
            camera matrix
        cam_dist : Mat
            distortion coefficients
        """
        allCorners = []
        allIds = []
        accepted = []
        rejected = []
        if minimum_views < 0:
            raise ValueError("minimum_views must be nonnegative")

        # Calibrate camera
        images = glob.glob(os.path.join(calibration_folder, self.image_pattern))
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                raise ValueError(f"Could not read calibration image: {fname}")
            if img.shape[1::-1] != (self.cam_width, self.cam_height):
                raise ValueError(
                    f"Calibration image size differs from camera size: {fname}"
                )
            if verbose:
                print("Reading image:", fname)

            # Convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, _, corners, ids = self.detect_markers(img, gray, draw=False)

            if ret:
                allCorners.append(corners)
                allIds.append(ids)
                accepted.append(fname)
            else:
                rejected.append(fname)
                if verbose:
                    print("Bad image:", fname)

        if len(allCorners) < minimum_views:
            raise ValueError(
                f"Camera calibration needs at least {minimum_views} valid views; "
                f"detected {len(allCorners)} in {len(images)} images"
            )
        if verbose:
            print("Camera calibration")
        if self.cam_mtx is None or self.cam_dist is None:
            cameraMatrixInit = np.array(
                [
                    [1000.0, 0.0, self.cam_width / 2.0],
                    [0.0, 1000.0, self.cam_height / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            distCoeffsInit = np.zeros((5, 1))
        else:
            cameraMatrixInit = self.cam_mtx
            distCoeffsInit = self.cam_dist

        effective_criteria = (
            criteria
            if criteria is not None
            else (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
        )
        ret, cam_mtx, cam_dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            allCorners,
            allIds,
            self.charuco_board,
            (self.cam_width, self.cam_height),
            cameraMatrixInit,
            distCoeffsInit,
            flags=calib_flags,
            criteria=effective_criteria,
        )
        per_view_rms = []
        board_points = self.charuco_board.getChessboardCorners()
        for corners, ids, rvec, tvec in zip(allCorners, allIds, rvecs, tvecs):
            projected, _ = cv2.projectPoints(
                board_points[np.asarray(ids).reshape(-1)], rvec, tvec, cam_mtx, cam_dist
            )
            residual = projected.reshape(-1, 2) - np.asarray(corners).reshape(-1, 2)
            per_view_rms.append(float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
        self.diagnostics = {
            "accepted_images": accepted,
            "rejected_images": rejected,
            "rms": float(ret),
            "per_view_rms": per_view_rms,
            "criteria": list(effective_criteria),
            "flags": int(calib_flags),
        }
        if verbose:
            print(cam_mtx)
            print(cam_dist)
            print("Error", ret)

        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

        return cam_mtx, cam_dist


class ProjectorCameraCalibrator:
    """Retain the reference projected-circle calibration workflow.

    Call ``get_circle_grid_image`` before ``calibrate`` to establish the
    projected circle coordinates. Defaults retain the historical 2560-by-1440
    stereo size and board. Both can be supplied explicitly for a different
    physical setup, without changing detector criteria or initialization.
    ``homography_reprojection_threshold`` is in destination board coordinates,
    not camera pixels; the default5.0 retains the source numeric value.
    """

    def __init__(
        self,
        cam_width=1920,
        cam_height=1080,
        cam_mtx=None,
        cam_dist=None,
        proj_width=1920,
        proj_height=1080,
        proj_mtx=None,
        proj_dist=None,
        circle_grid_size=(4, 11),
        circle_r=15,
        *,
        board: ChArUcoBoardConfig = ChArUcoBoardConfig(),
        stereo_image_size: tuple[int, int] = (2560, 1440),
        image_pattern: str = "*.jpg",
        homography_reprojection_threshold: float = 5.0,
    ):
        if (
            not isinstance(homography_reprojection_threshold, Real)
            or isinstance(homography_reprojection_threshold, bool)
            or not np.isfinite(float(homography_reprojection_threshold))
            or homography_reprojection_threshold <= 0
        ):
            raise ValueError(
                "homography_reprojection_threshold must be finite and positive"
            )
        self.homography_reprojection_threshold = float(
            homography_reprojection_threshold
        )

        # Define the charuco board parameters
        self.board_config = board
        self.stereo_image_size = stereo_image_size
        self.image_pattern = image_pattern
        self.dictionary = cv2.aruco.getPredefinedDictionary(board.dictionary_id)
        self.charuco_board = cv2.aruco.CharucoBoard(
            board.squares, board.square_length, board.marker_length, self.dictionary
        )
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        # Camera parameters (image size and intrinsic parameters)
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_mtx = cam_mtx
        self.cam_dist = cam_dist

        # Projector parameters (image size and intrinsic parameters)
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.proj_mtx = proj_mtx
        self.proj_dist = proj_dist

        # Generate circle grid
        self.circle_grid_size = circle_grid_size
        self.circle_r = circle_r
        self.circle_2d_pts = self.build_circle_grid_pts(circle_grid_size, circle_r)

    def detect_markers(self, frame, gray, draw=True):
        """
        Detect markers and corners of the charuco board in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected markers
        gray : Mat
            gray version of the image
        draw : bool, optional (default = True)
            draw detected markers or not

        Returns:
        -------
        ret : bool
            True if at least one marker and the board were detected
        frame : Mat
            Image with detected markers
        H : Mat
            Homography matrix of the camera to the charuco board
        """
        rvec = None
        tvec = None
        ret = False
        H = None
        # SUB PIXEL CORNER DETECTION CRITERION
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

        # Detect markers and corners
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )

        if corners is not None and len(corners) > 0:
            if draw:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, (3, 3), (-1, -1), criteria)

            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.charuco_board
            )
            if charuco_ids is not None and len(charuco_ids) > 0:
                if draw:
                    frame = cv2.aruco.drawDetectedCornersCharuco(
                        frame, charuco_corners, charuco_ids
                    )

                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    self.charuco_board,
                    self.cam_mtx,
                    self.cam_dist,
                    None,
                    None,
                )
                if ret and draw:
                    frame = cv2.drawFrameAxes(
                        frame, self.cam_mtx, self.cam_dist, rvec, tvec, 0.1
                    )

                board_indices = {
                    int(marker_id): index
                    for index, marker_id in enumerate(
                        self.charuco_board.getIds().ravel()
                    )
                }
                matched = [
                    (image_index, board_indices[int(marker_id)])
                    for image_index, marker_id in enumerate(ids.ravel())
                    if int(marker_id) in board_indices
                ]
                if matched:
                    # Keep detection order for known markers. Other markers in
                    # the same dictionary do not belong to this physical board.
                    image_indices, object_indices = zip(*matched)
                    obj_points = np.array(self.charuco_board.getObjPoints())[
                        list(object_indices), :, :2
                    ].reshape(-1, 2)
                    corners_flatten = np.array(corners)[list(image_indices)].reshape(
                        -1, 2
                    )
                    H, _ = cv2.findHomography(
                        corners_flatten,
                        obj_points,
                        cv2.RANSAC,
                        self.homography_reprojection_threshold,
                    )

        if not _valid_homography(H):
            return False, frame, None

        return ret, frame, H

    def detect_circle_grid(self, frame, gray, H, draw=True):
        """
        Detect the circle grid in the image

        Parameters:
        ---------
        frame : Mat
            image to draw detected circles grid
        gray : Mat
            gray version of the image
        H : Mat
            Homography matrix of the camera to the charuco board
        draw : bool, optional (default = True)
            draw detected circles or not

        Returns:
        -------
        ret : bool
            True if the circle grid was detected
        frame : Mat
            Image with detected circles
        circles : Mat
            2D coordinates of the detected circles in the camera
        circles3D : Mat
            3D coordinates of the detected circles in the charuco board reference frame (z=0)
        """
        if not _valid_homography(H):
            return False, frame, None, None
        ret, circles = cv2.findCirclesGrid(
            gray,
            self.circle_grid_size,
            flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        )

        circles3D = None
        if ret:
            if draw:
                frame = cv2.drawChessboardCorners(
                    frame, self.circle_grid_size, circles, ret
                )

            # Apply homography to get the 3D coordinates of the circles in the charuco board reference frame
            circles3D = cv2.perspectiveTransform(circles, H)
            # Add z=0 to the 3D coordinates
            circles3D = np.pad(
                circles3D, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=0
            )
            circles3D = circles3D.astype(np.float32)

        return ret, frame, circles, circles3D

    def build_circle_grid_pts(self, grid_size, circle_r):
        """
        Build the 2D coordinates of the circle grid

        Parameters:
        ---------
        grid_size : tuple
            Number of circles in the grid (nb_col, nb_row)
        circle_r : int
            Radius of the circles

        Returns:
        -------
        circle_2d_pts : Mat
            2D coordinates of the circles in the grid
        """
        nb_col, nb_row = grid_size
        circle_2d_pts = np.zeros((nb_col * nb_row, 2), dtype=np.int32)
        count = 0
        for i in range(nb_row - 1, -1, -1):
            for j in range(nb_col - 1, -1, -1):
                if i % 2 == 0:
                    pos_x = j * 6 * circle_r + (3 * circle_r)
                else:
                    pos_x = j * 6 * circle_r
                pos_y = i * 3 * circle_r
                circle_2d_pts[count] = [pos_x, pos_y]
                count += 1
        return circle_2d_pts

    def get_circle_grid_image(self, x, y):
        """
        Return an image of the circle grid

        Parameters:
        ---------
        x : int
            x position of the grid
        y : int
            y position of the grid

        Returns:
        -------
        img_circle : Mat
            Image of the circle grid
        """
        img_circle = np.zeros((self.proj_height, self.proj_width, 4), dtype=np.uint8)
        self.circle_2d = self.circle_2d_pts + [x, y]
        white_padding = 3 * self.circle_r
        min_x = np.min(self.circle_2d[:, 0]) - white_padding
        max_x = np.max(self.circle_2d[:, 0]) + white_padding
        min_y = np.min(self.circle_2d[:, 1]) - white_padding
        max_y = np.max(self.circle_2d[:, 1]) + white_padding
        img_circle[min_y:max_y, min_x:max_x] = cv2.bitwise_not(
            img_circle[min_y:max_y, min_x:max_x]
        )
        for c in self.circle_2d:
            img_circle = cv2.circle(
                img_circle,
                tuple(c.astype(np.int32)),
                self.circle_r,
                (0, 0, 0),
                cv2.FILLED,
            )
        return img_circle

    def calibrate(
        self,
        image_folder,
        calibrate_proj=True,
        *,
        minimum_views=0,
        verbose=True,
        projector_criteria=None,
        stereo_criteria=None,
    ):
        """
        Calibrate the camera and the projector using the charuco board and the circle grid

        Parameters:
        ---------
        image_folder : str
            path to the folder containing the calibration images
        calibrate_proj : bool, optional (default = True)
            calibrate the projector or not

        Returns:
        -------
        proj_mtx : Mat
            projector intrinsic parameters matrix
        proj_dist : Mat
            projector distortion coefficients
        proj_R : Mat
            rotation matrix between the camera and the projector
        proj_T : Mat
            translation vector between the camera and the projector
        R1 : Mat
            rectification transform (rotation matrix) for the camera
        R2 : Mat
            rectification transform (rotation matrix) for the projector
        P1 : Mat
            projection matrix in the new (rectified) coordinate systems for the camera
        P2 : Mat
            projection matrix in the new (rectified) coordinate systems for the projector
        Q : Mat
            disparity-to-depth mapping matrix
        """
        proj_obj_pts = []
        proj_circle_pts = []
        cam_circle_pts = []
        accepted = []
        rejected = []
        if minimum_views < 0:
            raise ValueError("minimum_views must be nonnegative")

        # Calibrate projector-camera stereo setup
        images = glob.glob(os.path.join(image_folder, self.image_pattern))
        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                raise ValueError(f"Could not read calibration image: {fname}")
            if img.shape[1::-1] != (self.cam_width, self.cam_height):
                raise ValueError(
                    f"Calibration image size differs from camera size: {fname}"
                )
            # Convert to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, img, H = self.detect_markers(img, gray)

            if ret and _valid_homography(H):
                # Find projected circles
                ret, img, circles_2d_cam, circles_3d = self.detect_circle_grid(
                    img, gray, H
                )
                if ret:
                    proj_obj_pts.append(circles_3d)
                    proj_circle_pts.append(
                        np.expand_dims(self.circle_2d.astype(np.float32), axis=1)
                    )
                    cam_circle_pts.append(circles_2d_cam)
                    accepted.append(fname)
                else:
                    rejected.append(fname)
                    if verbose:
                        print("Bad image:", fname)
            else:
                rejected.append(fname)

        if len(proj_obj_pts) < minimum_views:
            raise ValueError(
                f"Projector/stereo calibration needs at least {minimum_views} valid views; "
                f"detected {len(proj_obj_pts)} in {len(images)} images"
            )

        projector_rms = None
        if calibrate_proj:
            if self.proj_mtx is None:
                self.proj_mtx = np.array(
                    [
                        [3000.0, 0.0, self.proj_width / 2.0],
                        [0.0, 3000.0, self.proj_height / 2.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
            if verbose:
                print("\nProjector calibration")
            projector_options = {"flags": cv2.CALIB_USE_INTRINSIC_GUESS}
            if projector_criteria is not None:
                projector_options["criteria"] = projector_criteria
            ret, proj_mtx, proj_dist, _, _ = cv2.calibrateCamera(
                proj_obj_pts,
                proj_circle_pts,
                (self.proj_width, self.proj_height),
                self.proj_mtx,
                None,
                **projector_options,
            )
            projector_rms = float(ret)
            if verbose:
                print(proj_mtx)
                print(proj_dist)
                print("Error", ret)

            self.proj_mtx = proj_mtx
            self.proj_dist = proj_dist

        if verbose:
            print("\nStereo calibration")
        stereo_options = {"flags": cv2.CALIB_FIX_INTRINSIC}
        if stereo_criteria is not None:
            stereo_options["criteria"] = stereo_criteria
        error, cam_mtx, cam_dist, proj_mtx, proj_dist, proj_R, proj_T, _, _ = (
            cv2.stereoCalibrate(
                proj_obj_pts,
                cam_circle_pts,
                proj_circle_pts,
                self.cam_mtx,
                self.cam_dist,
                self.proj_mtx,
                self.proj_dist,
                self.stereo_image_size,
                **stereo_options,
            )
        )
        self.diagnostics = {
            "accepted_images": accepted,
            "rejected_images": rejected,
            "projector_rms": projector_rms,
            "stereo_rms": float(error),
            "stereo_image_size": list(self.stereo_image_size),
            "stereo_flags": int(cv2.CALIB_FIX_INTRINSIC),
            "homography_reprojection_threshold": self.homography_reprojection_threshold,
            "homography_reprojection_threshold_space": "board_coordinates",
            "projector_criteria": None
            if projector_criteria is None
            else list(projector_criteria),
            "stereo_criteria": None
            if stereo_criteria is None
            else list(stereo_criteria),
        }
        if verbose:
            print("Camera parameters")
            print(cam_mtx, cam_dist)
            print("Projector parameters")
            print(proj_mtx, proj_dist)
            print("Rotation and translation")
            print(proj_R)
            print(proj_T)
            print("Error", error)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cam_mtx,
            cam_dist,
            proj_mtx,
            proj_dist,
            (self.cam_width, self.cam_height),
            proj_R,
            proj_T,
        )

        return proj_mtx, proj_dist, proj_R, proj_T, R1, R2, P1, P2, Q
