"""Analytical geometry and output alignment without reference-source fixtures."""

import json

import cv2
import numpy as np
import pytest

from reossi.calibration.workflow import StereoCalibration
from reossi.plan import ScanPlan
from reossi.reconstruction.geometry import StereoTriangulator
from reossi.reconstruction.workflow import reconstruct_scan, write_ply


def known_scan(tmp_path, *, rotation=None, one_based=False):
    """Project known camera-frame points into a calibrated virtual projector."""
    camera_size = (3, 2)
    projector_size = (128, 96)
    camera_matrix = np.array([[100.0, 0, 1], [0, 100.0, 1], [0, 0, 1]])
    projector_matrix = np.array([[110.0, 0, 64], [0, 105.0, 48], [0, 0, 1]])
    rotation = np.eye(3) if rotation is None else rotation
    translation = np.array([[-0.12], [0.002], [0.005]])
    calibration = StereoCalibration(
        camera_matrix,
        np.zeros(5),
        camera_size,
        projector_matrix,
        np.zeros(5),
        projector_size,
        rotation,
        translation,
        length_unit="m",
        stereo_image_size=camera_size,
    )
    calibration_path = tmp_path / "stereo.npz"
    calibration.save(calibration_path)
    columns, rows = np.meshgrid(np.arange(3), np.arange(2))
    depth = np.array([[1.0, 0.4, 1.5], [0.6, 1.2, 0.8]])
    true_points = np.stack(
        ((columns - 1) * depth / 100, (rows - 1) * depth / 100, depth)
    )
    camera_points = true_points.reshape(3, -1)
    projected_points = rotation @ camera_points + translation
    pixels = projector_matrix @ projected_points
    pixels = (pixels[:2] / pixels[2]).reshape(2, 2, 3)
    folder = tmp_path / "scan"
    folder.mkdir()
    plan = ScanPlan(
        mode="reossi" if one_based else "conventional",
        projector_size=projector_size,
        emulated_size=(8, 6) if one_based else projector_size,
        repeats=1,
    )
    (folder / "scan.json").write_text(
        json.dumps({"format_version": 1, "plan": plan.to_dict()})
    )
    np.save(folder / "h_pixels.npy", pixels[0] + int(one_based))
    np.save(folder / "v_pixels.npy", pixels[1] + int(one_based))
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) * 10
    assert cv2.imwrite(str(folder / "white.png"), image)
    return folder, calibration_path, calibration, true_points, image


@pytest.mark.parametrize("coordinate_frame", ["camera", "projector", "legacy"])
def test_known_geometry_in_each_explicit_coordinate_frame(tmp_path, coordinate_frame):
    rotation, _ = cv2.Rodrigues(np.array([0.01, 0.04, -0.005]))
    folder, calibration_path, calibration, true_points, image = known_scan(
        tmp_path, rotation=rotation
    )
    result = reconstruct_scan(
        folder, calibration_path, tmp_path / "cloud", coordinate_frame=coordinate_frame
    )
    expected = true_points.reshape(3, -1)
    if coordinate_frame != "camera":
        expected = calibration.rotation @ expected
        if coordinate_frame == "projector":
            expected += calibration.translation
    # Correspondences retain the original column-major camera traversal.
    xy = result.camera_pixels.astype(int)
    linear_indices = xy[:, 1] * 3 + xy[:, 0]
    np.testing.assert_allclose(result.points, expected[:, linear_indices], atol=1e-5)
    np.testing.assert_array_equal(
        result.colors, image[xy[:, 1], xy[:, 0], ::-1] / 255.0
    )
    assert result.valid_mask.all()
    assert result.metadata["coordinate_frame"] == coordinate_frame
    assert result.metadata["length_unit"] == "m"
    np.testing.assert_array_equal(
        calibration.projector_matrix,
        np.array([[110.0, 0, 64], [0, 105.0, 48], [0, 0, 1]]),
    )


def test_box_filter_keeps_pixels_colors_and_mask_aligned(tmp_path):
    folder, calibration_path, _, true_points, image = known_scan(tmp_path)
    output = tmp_path / "filtered"
    result = reconstruct_scan(folder, calibration_path, output, box_threshold=0.9)
    expected_mask = true_points[2] < 0.9
    np.testing.assert_array_equal(result.valid_mask, expected_mask)
    assert result.points.shape == (3, 3)
    for index, (column, row) in enumerate(result.camera_pixels.astype(int)):
        np.testing.assert_allclose(
            result.points[:, index], true_points[:, row, column], atol=1e-5
        )
        np.testing.assert_array_equal(
            result.colors[index], image[row, column, ::-1] / 255.0
        )
    np.testing.assert_array_equal(np.load(output / "valid_mask.npy"), expected_mask)
    np.testing.assert_array_equal(
        np.load(output / "camera_pixels.npy"), result.camera_pixels
    )
    np.testing.assert_array_equal(
        np.load(output / "projector_pixels.npy"), result.projector_pixels
    )
    text = (output / "cloud.ply").read_text()
    assert "element vertex 3\n" in text
    assert "property uchar red\n" in text


def test_nonfinite_geometry_removes_the_same_correspondence_everywhere(
    tmp_path, monkeypatch
):
    folder, calibration_path, _, _, image = known_scan(tmp_path)
    original = StereoTriangulator.triangulate

    def with_one_invalid(self, camera_points, projector_points):
        result = original(self, camera_points, projector_points)
        result[:, 1] = np.nan
        return result

    monkeypatch.setattr(StereoTriangulator, "triangulate", with_one_invalid)
    result = reconstruct_scan(folder, calibration_path, tmp_path / "cloud")
    assert result.points.shape == (3, 5)
    assert not result.valid_mask[1, 0]
    assert result.metadata["nonfinite_points_removed"] == 1
    for index, (column, row) in enumerate(result.camera_pixels.astype(int)):
        np.testing.assert_array_equal(
            result.colors[index], image[row, column, ::-1] / 255.0
        )


def test_empty_cloud_is_a_valid_explicit_output(tmp_path):
    folder, calibration_path, _, _, _ = known_scan(tmp_path)
    np.save(folder / "h_pixels.npy", np.full((2, 3), -1))
    result = reconstruct_scan(folder, calibration_path, tmp_path / "empty")
    assert result.points.shape == (3, 0)
    assert result.colors.shape == (0, 3)
    assert result.camera_pixels.shape == result.projector_pixels.shape == (0, 2)
    assert not result.valid_mask.any()
    assert "element vertex 0\n" in (tmp_path / "empty/cloud.ply").read_text()


def test_one_based_conversion_is_explicit_and_recorded(tmp_path):
    folder, calibration_path, _, true_points, _ = known_scan(tmp_path, one_based=True)
    raw = reconstruct_scan(folder, calibration_path, tmp_path / "source")
    corrected = reconstruct_scan(
        folder, calibration_path, tmp_path / "zero_based", identifier_origin=1
    )
    np.testing.assert_allclose(
        raw.projector_pixels - corrected.projector_pixels, 1.0, atol=1e-5
    )
    xy = corrected.camera_pixels.astype(int)
    np.testing.assert_allclose(
        corrected.points, true_points[:, xy[:, 1], xy[:, 0]], atol=1e-5
    )
    assert raw.metadata["identifier_origin"] == "legacy"
    assert raw.metadata["plan_identifier_origin"] == 1
    assert corrected.metadata["identifier_origin"] == 1


def test_invalid_size_and_existing_output_are_rejected(tmp_path):
    folder, calibration_path, _, _, _ = known_scan(tmp_path)
    np.save(folder / "h_pixels.npy", np.zeros((1, 3)))
    with pytest.raises(ValueError, match="calibrated camera"):
        reconstruct_scan(folder, calibration_path, tmp_path / "cloud")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        reconstruct_scan(folder, calibration_path, existing)


def test_ply_checks_shape_finite_values_and_existing_files(tmp_path):
    path = tmp_path / "cloud.ply"
    write_ply(path, np.array([[1.0], [2.0], [3.0]]), np.array([[1.0, 0.5, 0.0]]))
    assert path.read_text().split("end_header\n")[1].split() == [
        "1",
        "2",
        "3",
        "255",
        "128",
        "0",
    ]
    with pytest.raises(FileExistsError):
        write_ply(path, np.empty((3, 0)), np.empty((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        write_ply(tmp_path / "bad.ply", np.full((3, 1), np.nan), np.zeros((1, 3)))


def test_point_guard_precedes_python_correspondence_collection(tmp_path, monkeypatch):
    folder, calibration_path, _, _, _ = known_scan(tmp_path)
    columns = np.load(folder / "h_pixels.npy")
    columns[1] = -1
    columns[0, 2] = np.nan
    np.save(folder / "h_pixels.npy", columns)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Candidate limit must be checked before allocating correspondence lists"
        )

    monkeypatch.setattr(StereoTriangulator, "collect_correspondences", forbidden)
    with pytest.raises(ValueError, match="2 valid candidate points.*max_points=1"):
        reconstruct_scan(folder, calibration_path, tmp_path / "limited", max_points=1)
    assert not (tmp_path / "limited").exists()


@pytest.mark.parametrize("max_points", [0, -1, float("nan"), float("inf"), 1.5, True])
def test_invalid_point_limit_fails_before_reading_scan(tmp_path, max_points):
    with pytest.raises(ValueError, match="max_points"):
        reconstruct_scan(
            tmp_path / "absent",
            tmp_path / "absent.npz",
            tmp_path / "output",
            max_points=max_points,
        )


def test_point_limit_can_be_disabled_explicitly_and_is_recorded(tmp_path):
    folder, calibration_path, _, _, _ = known_scan(tmp_path)
    result = reconstruct_scan(
        folder, calibration_path, tmp_path / "cloud", max_points=None
    )
    assert result.points.shape[1] == 6
    assert result.metadata["max_points"] is None
