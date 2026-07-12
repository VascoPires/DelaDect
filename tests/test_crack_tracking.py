import math

import numpy as np
import pytest

from deladect.detection.crack_tracking import CrackTrack, match_tracks, normalize_detections


def make_track():
    segment = np.array([[0.0, 0.0], [1.0, 1.0]])
    detection = normalize_detections([segment])[0]
    return CrackTrack(
        track_id=1,
        first_frame_abs=0,
        baseline_frame_abs=0,
        baseline_segment=segment,
        baseline_length_px=detection.length_px,
        baseline_bbox=detection.bbox,
        last_frame_abs=0,
        last_segment=segment,
        last_length_px=detection.length_px,
        last_bbox=detection.bbox,
    ), detection


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_center_px": 0, "max_angle_deg": 45, "max_cost": 1}, "max_center_px"),
        ({"max_center_px": -1, "max_angle_deg": 45, "max_cost": 1}, "max_center_px"),
        ({"max_center_px": math.inf, "max_angle_deg": 45, "max_cost": 1}, "max_center_px"),
        ({"max_center_px": 10, "max_angle_deg": 0, "max_cost": 1}, "max_angle_deg"),
        ({"max_center_px": 10, "max_angle_deg": math.nan, "max_cost": 1}, "max_angle_deg"),
        ({"max_center_px": 10, "max_angle_deg": 45, "max_cost": -1}, "max_cost"),
    ],
)
def test_match_tracks_rejects_invalid_thresholds(kwargs, message):
    track, detection = make_track()

    with pytest.raises(ValueError, match=message):
        match_tracks([track], [detection], **kwargs)
