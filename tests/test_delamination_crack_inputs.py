import numpy as np
import pytest

from deladect.detection.delamination import _coerce_cracks_by_frame, _result_key_token


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
