from types import SimpleNamespace

import pytest

from deladect.detection import crack_analysis
from deladect.detection import crack_detection
from deladect.specimen import Ply


def _ply(name, orientation, width=8.0, min_length=16.0):
    return Ply(
        name=name,
        orientation_deg=orientation,
        avg_crack_width_px=width,
        min_crack_length_px=min_length,
    )


def test_crack_analysis_evaluates_each_unique_layup_orientation_once(monkeypatch):
    specimen = SimpleNamespace(
        name="layup",
        plies=[
            _ply("zero-a", 0.0),
            _ply("ninety", 90.0),
            _ply("zero-b", 0.0),
        ],
    )
    evaluated = []

    def fake_crack_eval(_specimen, *, ply, **_kwargs):
        evaluated.append(ply)
        return {
            "cracks": [ply.name],
            "densities": [],
            "thresholds": [],
            "metrics": None,
            "paths": {},
            "params": {},
        }

    monkeypatch.setattr(crack_detection, "crack_eval", fake_crack_eval)

    results = crack_analysis(specimen)

    assert list(results) == ["0", "90"]
    assert [ply.name for ply in evaluated] == ["zero-a", "ninety"]
    assert [ply.name for ply in results["0"]["plies"]] == ["zero-a", "zero-b"]


@pytest.mark.parametrize(
    ("name", "invoke", "expected_orientations"),
    [
        (
            "crack_eval_by_orientation",
            lambda specimen: crack_detection.crack_eval_by_orientation(
                specimen,
                orientations=[10.0],
            ),
            [10.0],
        ),
        (
            "crack_eval_crossply",
            lambda specimen: crack_detection.crack_eval_crossply(specimen),
            [0.0, 90.0],
        ),
        (
            "crack_eval_plus_minus",
            lambda specimen: crack_detection.crack_eval_plus_minus(
                specimen,
                45.0,
                transverse_layer=True,
            ),
            [45.0, -45.0, 90.0],
        ),
    ],
)
def test_deprecated_crack_wrappers_warn_and_forward(
    monkeypatch,
    name,
    invoke,
    expected_orientations,
):
    captured = {}

    def fake_crack_analysis(specimen, **kwargs):
        captured["specimen"] = specimen
        captured.update(kwargs)
        return {"forwarded": {}}

    monkeypatch.setattr(crack_detection, "crack_analysis", fake_crack_analysis)
    specimen = object()

    with pytest.warns(DeprecationWarning, match=rf"{name}\(\) is deprecated"):
        result = invoke(specimen)

    assert result == {"forwarded": {}}
    assert captured["specimen"] is specimen
    assert captured["orientations"] == expected_orientations
