from types import SimpleNamespace

import numpy as np
import pytest

from deladect.detection import DiffuseDetector
from deladect.detection.delamination import (
    _coerce_cracks_by_frame,
    _crack_input_frame_count,
    _result_key_token,
)


def test_result_key_token_sanitizes_interface_display_name():
    assert _result_key_token("0/90") == "0_90"


def test_coerce_dense_crackdect_array_by_frame():
    cracks = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)

    frames = _coerce_cracks_by_frame(cracks, frame_count=2)

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], cracks[0])
    np.testing.assert_array_equal(frames[1], cracks[1])


def test_coerce_ragged_object_array_by_frame():
    cracks = np.empty(2, dtype=object)
    cracks[0] = np.zeros((1, 2, 2), dtype=float)
    cracks[1] = np.zeros((3, 2, 2), dtype=float)

    frames = _coerce_cracks_by_frame(cracks, frame_count=2)

    assert [frame.shape[0] for frame in frames] == [1, 3]


def test_coerce_segment_array_for_single_frame():
    cracks = np.zeros((3, 2, 2), dtype=float)

    frames = _coerce_cracks_by_frame(cracks, frame_count=1)

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], cracks)


def test_reject_ambiguous_segment_array_for_multiple_frames():
    cracks = np.zeros((3, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="only unambiguous for one frame"):
        _coerce_cracks_by_frame(cracks, frame_count=2)


def test_coerce_crack_analysis_result_merges_all_orientations_by_frame():
    zero_crack = np.array([[[0.0, 1.0], [2.0, 1.0]]])
    ninety_crack = np.array([[[3.0, 0.0], [3.0, 2.0]]])
    analysis = {
        "0": {"cracks": [zero_crack, np.empty((0, 2, 2))]},
        "90": {"cracks": [ninety_crack, ninety_crack]},
    }

    frames = _coerce_cracks_by_frame(analysis, frame_count=2)

    assert _crack_input_frame_count(analysis) == 2
    assert frames[0].shape == (2, 2, 2)
    assert frames[1].shape == (1, 2, 2)
    np.testing.assert_array_equal(frames[0][0], zero_crack[0])
    np.testing.assert_array_equal(frames[0][1], ninety_crack[0])


def test_coerce_filtered_crack_analysis_result_uses_present_orientation_only():
    cracks = [np.ones((1, 2, 2)), np.empty((0, 2, 2))]
    analysis = {"90": {"cracks": cracks}}

    frames = _coerce_cracks_by_frame(analysis, frame_count=2)

    np.testing.assert_array_equal(frames[0], cracks[0])
    assert frames[1].shape == (0, 2, 2)


def test_reject_crack_analysis_result_with_mismatched_frame_counts():
    analysis = {
        "0": {"cracks": [np.empty((0, 2, 2))]},
        "90": {"cracks": [np.empty((0, 2, 2)), np.empty((0, 2, 2))]},
    }

    with pytest.raises(ValueError, match="equal frame counts"):
        _coerce_cracks_by_frame(analysis, frame_count=2)


@pytest.mark.parametrize(
    ("analysis", "message"),
    [
        ({}, "at least one orientation"),
        ({"0": {}}, "containing a 'cracks' field"),
        ({"0": []}, "must be a mapping"),
        ({"0": {"cracks": []}}, "at least one crack frame"),
    ],
)
def test_reject_malformed_crack_analysis_results(analysis, message):
    with pytest.raises(ValueError, match=message):
        _coerce_cracks_by_frame(analysis, frame_count=1)


def test_standalone_diffuse_accepts_crack_analysis_result():
    owner = SimpleNamespace(
        specimen=SimpleNamespace(image_stack_full=None, avg_crack_width_px=8.0),
        save_preprocess_outputs=False,
        _uses_stack_overrides=lambda: False,
        _select_stacks=lambda: {"full": None},
        _kmeans_threshold=lambda values, fallback: fallback,
    )
    analysis = {
        "0": {"cracks": [np.array([[[2.0, 5.0], [17.0, 5.0]]])]},
        "90": {"cracks": [np.array([[[10.0, 2.0], [10.0, 17.0]]])]},
    }

    result = DiffuseDetector(owner).diffuse_delamination(
        cracks=analysis,
        processed_stack=[np.full((20, 20), 255, dtype=np.uint8)],
    )

    assert list(result["masks"]) == ["frame_0000"]
    assert result["masks"]["frame_0000"].shape == (20, 20)
