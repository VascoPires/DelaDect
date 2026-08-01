from types import SimpleNamespace

import numpy as np
import pytest

from deladect.detection.delamination import (
    EdgeDetector,
    _rebuild_edge_connected_columnwise,
    _rebuild_edge_connected_directional,
)


def test_directional_reconstruction_keeps_straight_edge_connected_path():
    candidate = np.zeros((6, 5), dtype=bool)
    candidate[:5, 2] = True

    rebuilt = _rebuild_edge_connected_directional(
        candidate, seed_depth=1, lateral_drift_px=1
    )

    np.testing.assert_array_equal(rebuilt, candidate)


def test_directional_reconstruction_allows_lateral_drift():
    candidate = np.zeros((5, 6), dtype=bool)
    candidate[0, 1] = True
    candidate[1, 2] = True
    candidate[2, 3] = True

    rebuilt = _rebuild_edge_connected_directional(
        candidate, seed_depth=1, lateral_drift_px=1
    )

    np.testing.assert_array_equal(rebuilt, candidate)


def test_directional_reconstruction_respects_drift_limit():
    candidate = np.zeros((4, 6), dtype=bool)
    candidate[0, 1] = True
    candidate[1:, 3] = True

    rebuilt = _rebuild_edge_connected_directional(
        candidate, seed_depth=1, lateral_drift_px=1
    )

    expected = np.zeros_like(candidate)
    expected[0, 1] = True
    np.testing.assert_array_equal(rebuilt, expected)


def test_directional_reconstruction_cannot_restart_below_a_gap():
    candidate = np.zeros((6, 4), dtype=bool)
    candidate[0:2, 1] = True
    candidate[3:, 1] = True

    rebuilt = _rebuild_edge_connected_directional(
        candidate, seed_depth=1, lateral_drift_px=1
    )

    expected = np.zeros_like(candidate)
    expected[0:2, 1] = True
    np.testing.assert_array_equal(rebuilt, expected)


def test_directional_reconstruction_requires_a_2d_mask():
    with pytest.raises(ValueError, match="2D mask"):
        _rebuild_edge_connected_directional(
            np.zeros((2, 3, 4), dtype=bool), seed_depth=1, lateral_drift_px=1
        )


def test_directional_is_the_default_connectivity_mode():
    detector = EdgeDetector(SimpleNamespace())

    resolved = detector._resolve_primary_params({})

    assert resolved["connectivity_mode"] == "directional"


def test_columnwise_mode_is_supported():
    detector = EdgeDetector(SimpleNamespace())

    resolved = detector._resolve_primary_params({"connectivity_mode": "columnwise"})

    assert resolved["connectivity_mode"] == "columnwise"


def test_columnwise_reconstruction_rejects_a_diagonal_step():
    candidate = np.zeros((4, 5), dtype=bool)
    candidate[0, 1] = True
    candidate[1:, 2] = True

    rebuilt = _rebuild_edge_connected_columnwise(candidate, seed_depth=1)

    expected = np.zeros_like(candidate)
    expected[0, 1] = True
    np.testing.assert_array_equal(rebuilt, expected)


def test_legacy_flood_connectivity_mode_has_targeted_migration_error():
    detector = EdgeDetector(SimpleNamespace())

    with pytest.raises(ValueError, match="legacy_flood.*removed"):
        detector._resolve_primary_params({"connectivity_mode": "legacy_flood"})
