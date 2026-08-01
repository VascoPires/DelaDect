import numpy as np
import pytest
from skimage.io import imsave

from deladect.specimen import Specimen


def _write_frames(folder, numbers, *, suffix="_sc"):
    folder.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        imsave(
            str(folder / f"{number:04d}{suffix}.png"),
            np.zeros((4, 4), dtype=np.uint8),
            check_contrast=False,
        )


def _make_region_specimen(tmp_path, *, full, upper, middle, lower, name="specimen"):
    data_root = tmp_path / "images"
    _write_frames(data_root / "full", full)
    _write_frames(data_root / "upper", upper)
    _write_frames(data_root / "middle", middle)
    _write_frames(data_root / "lower", lower)
    return Specimen(
        name=name,
        scale_px_mm=10.0,
        path_full=str(data_root / "full"),
        path_upper_border=str(data_root / "upper"),
        path_middle=str(data_root / "middle"),
        path_lower_border=str(data_root / "lower"),
        sorting_key="_sc",
        image_types=["png"],
        results_root=str(tmp_path / "results"),
    )


def test_aligned_regions_construct_successfully(tmp_path):
    frames = [2, 92, 189, 284, 378]
    specimen = _make_region_specimen(tmp_path, full=frames, upper=frames, middle=frames, lower=frames)

    assert len(specimen.image_stack_full) == 5
    assert len(specimen.image_stack_middle) == 5


def test_missing_frame_in_one_region_raises_named_error(tmp_path):
    full = [2, 92, 189, 284, 378]
    middle = [2, 92, 284, 378]  # missing frame 189

    with pytest.raises(ValueError, match=r"missing frame\(s\) \[189\]"):
        _make_region_specimen(tmp_path, full=full, upper=full, middle=middle, lower=full)


def test_extra_frame_in_one_region_raises_named_error(tmp_path):
    full = [2, 92, 189, 284, 378]
    upper = [2, 92, 189, 284, 378, 500]  # extra frame 500

    with pytest.raises(ValueError, match=r"extra frame\(s\) \[500\]"):
        _make_region_specimen(tmp_path, full=full, upper=upper, middle=full, lower=full)


def test_reordered_frames_across_regions_are_still_detected(tmp_path):
    # sort_paths always returns frames in ascending numeric order, so a region
    # cannot actually be "out of order" on disk when the same sorting_key is used;
    # this instead covers the same-count/different-identity edge case.
    full = [2, 92, 189, 284, 378]
    lower = [2, 92, 189, 284, 999]  # same count, one frame swapped for a different id

    with pytest.raises(ValueError, match="Frame alignment mismatch"):
        _make_region_specimen(tmp_path, full=full, upper=full, middle=full, lower=lower)
