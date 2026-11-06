"""Calibration contracts and real OpenCV solves on small known observations."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from reossi.calibration import observations
from reossi.calibration.solvers import (
    CameraCalibrator,
    ChArUcoBoardConfig,
    ProjectorCameraCalibrator,
)
from reossi.calibration.workflow import (
    CameraCalibration,
    calibrate_camera_images,
    calibrate_projector_images,
    load_camera_calibration,
    load_stereo_calibration,
    save_board_image,
    save_circle_grid,
)


def camera_parameters(image_size=(640, 480)):
    return CameraCalibration(
        np.array(
            [[600.0, 0, image_size[0] / 2], [0, 610.0, image_size[1] / 2], [0, 0, 1]]
        ),
        np.zeros(5),
        image_size,
        rms=0.0,
    )


def poses():
    for index in range(8):
        yield (
            np.array([0.08 + index * 0.035, -0.15 + index * 0.04, -0.1 + index * 0.02]),
            np.array(
                [-0.1 + index * 0.01, -0.12 + (index % 3) * 0.025, 0.7 + index * 0.07]
            ),
        )


def image_inputs(folder, size, count):
    folder.mkdir()
    for index in range(count):
        image = np.full((size[1], size[0], 3), index, dtype=np.uint8)
        assert cv2.imwrite(str(folder / f"view_{index}.png"), image)


def test_camera_workflow_recovers_projected_intrinsics(tmp_path, monkeypatch):
    expected = camera_parameters()
    board = CameraCalibrator().charuco_board
    world_points = board.getChessboardCorners()
    corner_ids = np.arange(len(world_points), dtype=np.int32)[:, None]
    projected_views = [
        cv2.projectPoints(world_points, r, t, expected.matrix, expected.distortion)[0]
        for r, t in poses()
    ]
    folder = tmp_path / "images"
    image_inputs(folder, expected.image_size, len(projected_views))

    def detections(self, frame, gray, draw=False):
        return (
            len(corner_ids),
            frame,
            projected_views[int(frame[0, 0, 0])],
            corner_ids.copy(),
        )

    monkeypatch.setattr(CameraCalibrator, "detect_markers", detections)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-10)
    result = calibrate_camera_images(
        folder,
        tmp_path / "camera.npz",
        image_size=expected.image_size,
        image_pattern="*.png",
        criteria=criteria,
    )
    np.testing.assert_allclose(result.matrix, expected.matrix, atol=0.01)
    assert result.rms < 0.0001
    assert len(result.diagnostics["accepted_images"]) == 8
    assert max(result.diagnostics["per_view_rms"]) < 0.0001
    assert result.diagnostics["criteria"] == list(criteria)
    loaded = load_camera_calibration(tmp_path / "camera.npz")
    np.testing.assert_array_equal(loaded.matrix, result.matrix)
    assert loaded.board == expected.board
    assert loaded.length_unit == "m"


@pytest.mark.parametrize("length_scale", [1.0, 1000.0])
def test_projector_stereo_workflow_recovers_known_geometry(
    tmp_path, monkeypatch, length_scale
):
    camera = camera_parameters()
    board = ChArUcoBoardConfig(
        squares=(10, 7),
        square_length=0.045 * length_scale,
        marker_length=0.035 * length_scale,
        dictionary_id=cv2.aruco.DICT_4X4_100,
    )
    length_unit = "m" if length_scale == 1 else "mm"
    projector_matrix = np.array([[720.0, 0, 400], [0, 700.0, 300], [0, 0, 1]])
    rotation, _ = cv2.Rodrigues(np.array([0.005, 0.03, -0.002]))
    translation = np.array([[-0.12], [0.002], [0.001]])
    grid = ProjectorCameraCalibrator(
        circle_grid_size=(4, 5), circle_r=8
    ).circle_2d_pts + [300, 200]
    rays = (
        np.linalg.inv(projector_matrix) @ np.column_stack((grid, np.ones(len(grid)))).T
    )
    views = []
    for rvec, tvec in poses():
        board_rotation, _ = cv2.Rodrigues(rvec)
        board_origin = tvec.reshape(3, 1)
        plane_normal = rotation @ board_rotation[:, 2]
        plane_origin = rotation @ board_origin + translation
        distances = float(plane_normal @ plane_origin[:, 0]) / (plane_normal @ rays)
        projector_points = rays * distances
        camera_points = rotation.T @ (projector_points - translation)
        object_points = (board_rotation.T @ (camera_points - board_origin)).T
        object_points[:, 2] = 0
        camera_pixels = camera.matrix @ camera_points
        camera_pixels = (camera_pixels[:2] / camera_pixels[2]).T.reshape(-1, 1, 2)
        views.append(
            (
                camera_pixels.astype(np.float32),
                (object_points * length_scale).astype(np.float32).reshape(-1, 1, 3),
            )
        )
    folder = tmp_path / "images"
    image_inputs(folder, camera.image_size, len(views))

    def markers(self, frame, gray, draw=True):
        return True, frame, np.eye(3)

    def circles(self, frame, gray, homography, draw=True):
        camera_pixels, object_points = views[int(frame[0, 0, 0])]
        return True, frame, camera_pixels, object_points

    monkeypatch.setattr(ProjectorCameraCalibrator, "detect_markers", markers)
    monkeypatch.setattr(ProjectorCameraCalibrator, "detect_circle_grid", circles)
    source_default = calibrate_projector_images(
        folder,
        tmp_path / "stereo-default.npz",
        camera,
        projector_size=(800, 600),
        grid_size=(4, 5),
        circle_radius=8,
        grid_origin=(300, 200),
        image_pattern="*.png",
        stereo_image_size=camera.image_size,
        board=board,
        length_unit=length_unit,
    )
    # The inherited 3000px focal guess needs more than OpenCV's default
    # iterations on these views. This preserves the observed source limitation
    # instead of silently changing its termination criterion.
    assert source_default.projector_rms > 0.1
    assert abs(source_default.projector_matrix[0, 0] - projector_matrix[0, 0]) > 100
    assert source_default.diagnostics["projector_criteria"] is None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-12)
    result = calibrate_projector_images(
        folder,
        tmp_path / "stereo.npz",
        camera,
        projector_size=(800, 600),
        grid_size=(4, 5),
        circle_radius=8,
        grid_origin=(300, 200),
        image_pattern="*.png",
        stereo_image_size=camera.image_size,
        projector_criteria=criteria,
        board=board,
        length_unit=length_unit,
        homography_reprojection_threshold=0.001 * length_scale,
    )
    np.testing.assert_allclose(result.projector_matrix, projector_matrix, atol=0.05)
    np.testing.assert_allclose(result.rotation, rotation, atol=1e-5)
    np.testing.assert_allclose(
        result.translation, translation * length_scale, atol=1e-5 * length_scale
    )
    assert result.projector_rms < 0.0001
    assert result.stereo_rms < 0.0001
    assert result.stereo_image_size == camera.image_size
    assert result.diagnostics["grid_origin"] == [300, 200]
    assert result.diagnostics["projector_criteria"] == list(criteria)
    assert source_default.diagnostics["homography_reprojection_threshold"] == 5.0
    assert (
        result.diagnostics["homography_reprojection_threshold"] == 0.001 * length_scale
    )
    assert (
        result.diagnostics["homography_reprojection_threshold_space"]
        == "board_coordinates"
    )
    assert result.diagnostics["homography_reprojection_threshold_unit"] == length_unit
    loaded = load_stereo_calibration(tmp_path / "stereo.npz")
    np.testing.assert_array_equal(loaded.rotation, result.rotation)
    np.testing.assert_array_equal(loaded.translation, result.translation)
    assert loaded.length_unit == length_unit
    assert loaded.board == board
    assert (
        loaded.diagnostics["homography_reprojection_threshold"] == 0.001 * length_scale
    )
    assert camera.length_unit == "m"
    assert camera.board == ChArUcoBoardConfig()


def test_camera_workflow_rejects_empty_or_unusable_images_before_solver(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        cv2.aruco,
        "calibrateCameraCharuco",
        lambda *args, **kwargs: pytest.fail("No valid observations"),
    )
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(ValueError, match="at least 3 valid views"):
        calibrate_camera_images(folder, tmp_path / "camera.npz", image_size=(64, 48))
    assert not (tmp_path / "camera.npz").exists()
    assert cv2.imwrite(str(folder / "blank.jpg"), np.zeros((48, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="detected 0 in 1"):
        calibrate_camera_images(folder, tmp_path / "camera.npz", image_size=(64, 48))


@pytest.mark.parametrize("case", ["unreadable", "wrong_size"])
def test_camera_workflow_fails_clearly_on_invalid_images(tmp_path, case):
    folder = tmp_path / "images"
    folder.mkdir()
    if case == "unreadable":
        (folder / "broken.jpg").write_bytes(b"not an image")
        message = "Could not read"
    else:
        assert cv2.imwrite(
            str(folder / "small.jpg"), np.zeros((12, 16, 3), dtype=np.uint8)
        )
        message = "image size differs"
    with pytest.raises(ValueError, match=message):
        calibrate_camera_images(folder, tmp_path / "camera.npz", image_size=(64, 48))


def test_board_and_projected_grid_generation_are_real_detectable_images(tmp_path):
    board_path = save_board_image(tmp_path / "board.png", image_size=(500, 700))
    image = cv2.imread(str(board_path))
    solver = CameraCalibrator()
    valid, _, corners, ids = solver.detect_markers(
        image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), draw=False
    )
    assert valid
    assert len(corners) == len(ids) == 24
    grid_path = save_circle_grid(
        tmp_path / "grid.png",
        projector_size=(640, 480),
        grid_size=(4, 5),
        circle_radius=8,
        grid_origin=(200, 150),
    )
    assert cv2.imread(str(grid_path), cv2.IMREAD_UNCHANGED).shape == (480, 640, 4)
    with pytest.raises(ValueError, match="must fit"):
        save_circle_grid(tmp_path / "offscreen.png", projector_size=(100, 100))


def test_calibration_file_protects_existing_output_and_validates_parameters(tmp_path):
    calibration = camera_parameters()
    path = tmp_path / "camera.npz"
    calibration.save(path)
    with pytest.raises(FileExistsError):
        calibration.save(path)
    with pytest.raises(ValueError, match="focal"):
        CameraCalibration(np.diag([-1, 1, 1]), np.zeros(5), (64, 48))
    with pytest.raises(ValueError, match="image_size"):
        CameraCalibration(np.eye(3), np.zeros(5), (64, 0))
    with pytest.raises(ValueError, match="board lengths|Board lengths"):
        ChArUcoBoardConfig(marker_length=0.05)


def test_projector_workflow_requires_a_physical_board(tmp_path):
    camera = camera_parameters()
    camera.board = None
    with pytest.raises(ValueError, match="physical ChArUco board"):
        calibrate_projector_images(
            tmp_path,
            tmp_path / "stereo.npz",
            camera,
            projector_size=(1920, 1080),
        )


def _mock_windows(monkeypatch, key=ord("c")):
    closed = []
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: key)
    monkeypatch.setattr(cv2, "destroyWindow", closed.append)
    return closed


def test_camera_observation_capture_saves_original_and_leaves_camera_owned_by_caller(
    tmp_path, monkeypatch
):
    closed = _mock_windows(monkeypatch)
    frame = np.full((16, 20, 3), 80, dtype=np.uint8)
    camera = SimpleNamespace(get_frame=lambda: frame.copy())

    def detector(self, annotated, gray, draw=True):
        annotated[:] = 255
        return True, annotated, [], []

    monkeypatch.setattr(CameraCalibrator, "detect_markers", detector)
    paths = observations.capture_camera_calibration_images(
        camera, tmp_path, max_images=1, extension=".png", preview_size=(20, 16)
    )
    assert len(paths) == 1
    np.testing.assert_array_equal(cv2.imread(str(paths[0])), frame)
    assert closed == ["REOSSI calibration camera"]


def test_projector_autocapture_uses_six_detections_and_cleans_up(tmp_path, monkeypatch):
    closed = _mock_windows(monkeypatch, key=-1)
    detections = []
    frame = np.full((48, 64, 3), 80, dtype=np.uint8)
    camera = SimpleNamespace(get_frame=lambda: frame.copy())
    calibration = camera_parameters((64, 48))

    def markers(image, gray, draw=True):
        detections.append(True)
        return True, image, np.eye(3)

    solver = SimpleNamespace(
        get_circle_grid_image=lambda *args: frame,
        detect_markers=markers,
        detect_circle_grid=lambda image, gray, h, draw=True: (True, image, None, None),
    )
    monkeypatch.setattr(
        observations, "_circle_calibrator", lambda *args, **kwargs: solver
    )
    paths = observations.capture_projector_calibration_images(
        camera,
        tmp_path,
        calibration,
        projector_size=(640, 480),
        max_images=1,
        extension=".png",
        preview_size=(64, 48),
    )
    assert len(paths) == 1
    assert len(detections) == 6
    assert closed == ["REOSSI calibration projector", "REOSSI calibration camera"]


def test_observation_capture_closes_windows_on_detector_failure(tmp_path, monkeypatch):
    closed = _mock_windows(monkeypatch)
    camera = SimpleNamespace(get_frame=lambda: np.zeros((16, 20, 3), dtype=np.uint8))

    def failed(*args, **kwargs):
        raise ValueError("bad observation")

    monkeypatch.setattr(CameraCalibrator, "detect_markers", failed)
    with pytest.raises(ValueError, match="bad observation"):
        observations.capture_camera_calibration_images(camera, tmp_path)
    assert closed == ["REOSSI calibration camera"]


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf"), True, "5"])
def test_homography_threshold_is_validated_before_board_allocation(
    threshold, monkeypatch
):
    monkeypatch.setattr(
        cv2.aruco,
        "getPredefinedDictionary",
        lambda *args: pytest.fail("Invalid threshold reached board allocation"),
    )
    with pytest.raises(ValueError, match="homography_reprojection_threshold"):
        ProjectorCameraCalibrator(homography_reprojection_threshold=threshold)


@pytest.mark.parametrize("threshold", [None, 0.002])
def test_marker_homography_uses_explicit_threshold_and_retains_source_default(
    threshold, monkeypatch
):
    options = (
        {} if threshold is None else {"homography_reprojection_threshold": threshold}
    )
    solver = ProjectorCameraCalibrator(
        cam_mtx=np.eye(3), cam_dist=np.zeros(5), **options
    )
    corner = np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]], dtype=np.float32)
    monkeypatch.setattr(
        cv2.aruco, "detectMarkers", lambda *a, **k: ([corner], np.array([[0]]), [])
    )
    monkeypatch.setattr(cv2, "cornerSubPix", lambda image, corners, *a: corners)
    monkeypatch.setattr(
        cv2.aruco,
        "interpolateCornersCharuco",
        lambda *a: (1, np.array([[[15, 15]]], dtype=np.float32), np.array([[0]])),
    )
    monkeypatch.setattr(
        cv2.aruco,
        "estimatePoseCharucoBoard",
        lambda *a: (True, np.zeros((3, 1)), np.ones((3, 1))),
    )
    calls = []

    def homography(source, destination, method, reprojection_threshold):
        calls.append(
            (source.copy(), destination.copy(), method, reprojection_threshold)
        )
        return np.eye(3), np.ones((len(source), 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "findHomography", homography)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    valid, _, transform = solver.detect_markers(image, image[:, :, 0], draw=False)
    assert valid
    np.testing.assert_array_equal(transform, np.eye(3))
    assert len(calls) == 1
    source, destination, method, selected = calls[0]
    np.testing.assert_array_equal(source, corner.reshape(4, 2))
    np.testing.assert_array_equal(
        destination, solver.charuco_board.getObjPoints()[0][:, :2]
    )
    assert method == cv2.RANSAC
    assert selected == (5.0 if threshold is None else threshold)


def test_homography_destination_units_require_converting_the_threshold():
    columns, rows = np.meshgrid(np.linspace(20, 620, 5), np.linspace(20, 460, 5))
    camera_pixels = np.column_stack((columns.ravel(), rows.ravel()))
    known = np.array(
        [[0.0008, 0.00003, 0.02], [-0.00002, 0.0007, 0.01], [0.0001, 0.0002, 1]]
    )
    true_board_metres = cv2.perspectiveTransform(camera_pixels[:, None, :], known)[
        :, 0, :
    ]
    observed_board_metres = true_board_metres.copy()
    outliers = np.array([1, 7, 13, 19, 23])
    observed_board_metres[outliers] += [0.04, -0.03]

    metric_h, metric_mask = cv2.findHomography(
        camera_pixels, observed_board_metres, cv2.RANSAC, 0.0005
    )
    millimetre_h, millimetre_mask = cv2.findHomography(
        camera_pixels, observed_board_metres * 1000, cv2.RANSAC, 0.5
    )
    expected_inliers = np.ones(25, dtype=np.uint8)
    expected_inliers[outliers] = 0
    np.testing.assert_array_equal(metric_mask[:, 0], expected_inliers)
    np.testing.assert_array_equal(millimetre_mask, metric_mask)
    metric_prediction = cv2.perspectiveTransform(camera_pixels[:, None, :], metric_h)[
        :, 0, :
    ]
    millimetre_prediction = (
        cv2.perspectiveTransform(camera_pixels[:, None, :], millimetre_h)[:, 0, :]
        / 1000
    )
    np.testing.assert_allclose(metric_prediction, true_board_metres, atol=1e-7)
    np.testing.assert_allclose(millimetre_prediction, metric_prediction, atol=1e-7)

    # Retaining the numeric source default is not a unit conversion: 5m and
    # 5mm select different inliers on this deliberately controlled fixture.
    _, five_metres = cv2.findHomography(
        camera_pixels, observed_board_metres, cv2.RANSAC, 5.0
    )
    _, five_millimetres = cv2.findHomography(
        camera_pixels, observed_board_metres * 1000, cv2.RANSAC, 5.0
    )
    assert five_metres.sum() == 25
    assert five_millimetres.sum() == 20


def test_projector_collection_rejects_threshold_before_opening_camera(
    tmp_path, monkeypatch, capsys
):
    from reossi.cli.main import main

    calibration = tmp_path / "camera.npz"
    camera_parameters((64, 48)).save(calibration)
    config = tmp_path / "calibration.json"
    config.write_text(
        json.dumps(
            {
                "camera_size": [64, 48],
                "projector_size": [640, 480],
                "circle_grid": {"size": [4, 5], "radius": 8, "origin": [200, 150]},
                "homography_reprojection_threshold": 0,
            }
        )
    )
    monkeypatch.setattr(
        "reossi.cli.calibrate.open_camera",
        lambda **kwargs: pytest.fail("Invalid settings opened a camera"),
    )
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "collect-projector-calibration",
                "--config",
                str(config),
                "--camera-calibration",
                str(calibration),
                "--output",
                str(tmp_path / "observations"),
            ]
        )
    assert "homography_reprojection_threshold" in capsys.readouterr().err


def test_projector_cli_forwards_manual_mode_and_threshold_without_hardware(
    tmp_path, monkeypatch, capsys
):
    from reossi.cli.main import main

    calibration_path = tmp_path / "camera.npz"
    camera_parameters((64, 48)).save(calibration_path)
    config = tmp_path / "calibration.json"
    config.write_text(
        json.dumps(
            {
                "camera_size": [64, 48],
                "projector_size": [640, 480],
                "circle_grid": {"size": [4, 5], "radius": 8, "origin": [200, 150]},
                "homography_reprojection_threshold": 0.001,
                "acquisition": {"consecutive_valid_frames": 0},
            }
        )
    )
    events = []
    camera = SimpleNamespace(stop_cam=lambda: events.append("stopped"))
    monkeypatch.setattr("reossi.cli.calibrate.open_camera", lambda **kwargs: camera)

    def capture(selected_camera, output, calibration, **kwargs):
        assert selected_camera is camera
        assert calibration.image_size == (64, 48)
        assert kwargs["consecutive_valid_frames"] == 0
        assert kwargs["homography_reprojection_threshold"] == 0.001
        events.append("capture")
        return []

    monkeypatch.setattr(
        "reossi.cli.calibrate.capture_projector_calibration_images", capture
    )
    assert (
        main(
            [
                "collect-projector-calibration",
                "--config",
                str(config),
                "--camera-calibration",
                str(calibration_path),
                "--output",
                str(tmp_path / "observations"),
            ]
        )
        == 0
    )
    assert events == ["capture", "stopped"]
    assert json.loads(capsys.readouterr().out)["observations"] == 0


@pytest.mark.parametrize(
    "command", ["collect-camera-calibration", "collect-projector-calibration"]
)
@pytest.mark.parametrize(
    "primary_error_type",
    [RuntimeError, KeyboardInterrupt, None],
    ids=["collection-fails", "collection-interrupted", "close-only-fails"],
)
def test_collection_preserves_primary_error_and_reports_close_only_failure(
    tmp_path, monkeypatch, caplog, command, primary_error_type
):
    from reossi.cli import calibrate
    from reossi.cli.main import build_parser

    calibration = tmp_path / "camera.npz"
    camera_parameters((64, 48)).save(calibration)
    config = tmp_path / "calibration.json"
    config.write_text(
        json.dumps(
            {
                "camera_size": [64, 48],
                "projector_size": [640, 480],
                "circle_grid": {"size": [4, 5], "radius": 8, "origin": [200, 150]},
            }
        )
    )
    output = tmp_path / "observations"
    completed = output / "completed-observation.txt"
    primary_error = (
        primary_error_type("Primary collection failure")
        if primary_error_type is not None
        else None
    )
    close_error = TimeoutError("Mock camera reader is still blocked")
    events = []

    def stop():
        events.append("stop")
        raise close_error

    camera = SimpleNamespace(stop_cam=stop)
    monkeypatch.setattr(calibrate, "open_camera", lambda **kwargs: camera)

    def capture(selected_camera, folder, *args, **kwargs):
        assert selected_camera is camera
        assert folder == output
        events.append("capture")
        folder.mkdir()
        completed.write_text("Previously completed observation is retained")
        if primary_error is not None:
            raise primary_error
        return [completed]

    monkeypatch.setattr(calibrate, "capture_camera_calibration_images", capture)
    monkeypatch.setattr(calibrate, "capture_projector_calibration_images", capture)
    arguments = [command, "--config", str(config), "--output", str(output)]
    if command == "collect-projector-calibration":
        arguments += ["--camera-calibration", str(calibration)]
    args = build_parser().parse_args(arguments)
    expected = primary_error if primary_error is not None else close_error
    with pytest.raises(type(expected)) as raised:
        args.handler(args)
    assert raised.value is expected
    assert events == ["capture", "stop"]
    assert completed.read_text() == "Previously completed observation is retained"
    close_records = [
        record for record in caplog.records if record.name == calibrate.__name__
    ]
    if primary_error is not None:
        assert len(close_records) == 1
        assert close_records[0].levelname == "ERROR"
        assert (
            close_records[0].getMessage()
            == "Camera shutdown also failed during collection"
        )
        assert close_records[0].exc_info[1] is close_error
        assert "Mock camera reader is still blocked" in caplog.text
    else:
        assert close_records == []


@pytest.mark.parametrize("custom_marker_ids", [False, True])
def test_real_board_homography_ignores_foreign_marker_in_same_dictionary(
    custom_marker_ids,
):
    matrix = np.array([[1000.0, 0, 400], [0, 1000.0, 500], [0, 0, 1]])
    solver = ProjectorCameraCalibrator(
        cam_width=800, cam_height=1000, cam_mtx=matrix, cam_dist=np.zeros(5)
    )
    if custom_marker_ids:
        solver.charuco_board = cv2.aruco.CharucoBoard(
            (5, 7), 0.04, 0.02, solver.dictionary, np.arange(40, 57, dtype=np.int32)
        )
    plain = np.full((1000, 800), 255, dtype=np.uint8)
    plain[20:720, 20:520] = solver.charuco_board.generateImage(
        (500, 700), marginSize=20
    )
    with_foreign = plain.copy()
    with_foreign[100:220, 600:720] = cv2.aruco.generateImageMarker(
        solver.dictionary, 99, 120
    )
    corners, ids, _ = cv2.aruco.detectMarkers(
        with_foreign, solver.dictionary, parameters=solver.params
    )
    assert 99 in ids
    board_ids = set(solver.charuco_board.getIds().ravel())
    assert 99 not in board_ids
    assert board_ids.issubset(ids.ravel())
    count, _, _ = cv2.aruco.interpolateCornersCharuco(
        corners, ids, with_foreign, solver.charuco_board
    )
    assert count == 24
    plain_valid, _, plain_h = solver.detect_markers(
        cv2.cvtColor(plain, cv2.COLOR_GRAY2BGR), plain, draw=False
    )
    foreign_valid, _, foreign_h = solver.detect_markers(
        cv2.cvtColor(with_foreign, cv2.COLOR_GRAY2BGR), with_foreign, draw=False
    )
    assert plain_valid and foreign_valid
    assert np.isfinite(foreign_h).all()
    np.testing.assert_array_equal(foreign_h, plain_h)


def test_degenerate_marker_geometry_returns_false_and_rejects_calibration_view(
    tmp_path, monkeypatch
):
    camera = camera_parameters((64, 64))
    solver = ProjectorCameraCalibrator(
        cam_width=64, cam_height=64, cam_mtx=camera.matrix, cam_dist=camera.distortion
    )
    # Detection/pose are controlled, but the homography solver sees genuinely
    # collinear camera points and returns None rather than an invertible map.
    collapsed = np.array([[[10, 10], [20, 10], [30, 10], [40, 10]]], dtype=np.float32)
    monkeypatch.setattr(
        cv2.aruco, "detectMarkers", lambda *a, **k: ([collapsed], np.array([[0]]), [])
    )
    monkeypatch.setattr(cv2, "cornerSubPix", lambda image, points, *a: points)
    monkeypatch.setattr(
        cv2.aruco,
        "interpolateCornersCharuco",
        lambda *a: (1, np.array([[[15, 10]]], dtype=np.float32), np.array([[0]])),
    )
    monkeypatch.setattr(
        cv2.aruco,
        "estimatePoseCharucoBoard",
        lambda *a: (True, np.zeros((3, 1)), np.ones((3, 1))),
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    valid, _, homography = solver.detect_markers(image, image[:, :, 0], draw=False)
    assert not valid
    assert homography is None
    monkeypatch.setattr(
        cv2,
        "findCirclesGrid",
        lambda *a, **k: pytest.fail("Invalid homography reached circle detection"),
    )
    folder = tmp_path / "images"
    folder.mkdir()
    assert cv2.imwrite(str(folder / "degenerate.jpg"), image)
    with pytest.raises(ValueError, match="detected 0 in 1 images"):
        calibrate_projector_images(
            folder,
            tmp_path / "stereo.npz",
            camera,
            projector_size=(640, 480),
            grid_size=(4, 5),
            circle_radius=8,
            grid_origin=(200, 150),
            minimum_views=1,
        )
    assert not (tmp_path / "stereo.npz").exists()


@pytest.mark.parametrize(
    "homography",
    [None, np.zeros((2, 3)), np.full((3, 3), np.nan), np.full((3, 3), np.inf)],
)
def test_circle_detection_rejects_missing_or_invalid_homography_early(
    homography, monkeypatch
):
    monkeypatch.setattr(
        cv2,
        "findCirclesGrid",
        lambda *a, **k: pytest.fail("Invalid map reached OpenCV"),
    )
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    valid, returned, circles, objects = ProjectorCameraCalibrator().detect_circle_grid(
        image, image[:, :, 0], homography, draw=False
    )
    assert not valid
    assert returned is image
    assert circles is None and objects is None


def test_batch_calibration_rejects_missing_homography_even_if_detector_reports_success(
    tmp_path, monkeypatch
):
    camera = camera_parameters((64, 48))
    folder = tmp_path / "images"
    image_inputs(folder, camera.image_size, 1)
    monkeypatch.setattr(
        ProjectorCameraCalibrator,
        "detect_markers",
        lambda self, frame, gray: (True, frame, None),
    )
    monkeypatch.setattr(
        ProjectorCameraCalibrator,
        "detect_circle_grid",
        lambda *a, **k: pytest.fail("Missing map reached circle geometry"),
    )
    with pytest.raises(ValueError, match="detected 0 in 1 images"):
        calibrate_projector_images(
            folder,
            tmp_path / "stereo.npz",
            camera,
            projector_size=(640, 480),
            grid_size=(4, 5),
            circle_radius=8,
            grid_origin=(200, 150),
            image_pattern="*.png",
            minimum_views=1,
        )
