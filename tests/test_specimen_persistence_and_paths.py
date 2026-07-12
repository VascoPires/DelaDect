import json

import pytest

from deladect.specimen import Ply, Specimen


def make_specimen(tmp_path, **overrides):
    options = {
        "name": "sample",
        "scale_px_mm": 10.0,
        "path_full": "unused",
        "sorting_key": "frame",
        "image_types": [".png"],
        "auto_init_stacks": False,
        "results_root": str(tmp_path / "results"),
    }
    options.update(overrides)
    return Specimen(**options)


def test_specimen_creation_and_mutation_do_not_write_config(tmp_path):
    specimen = make_specimen(tmp_path)
    assert not specimen.results_root_path().exists()

    config_path = specimen.config_path()
    assert not config_path.exists()
    specimen.add_ply(name="0-degree", orientation_deg=0.0)
    specimen.add_interface(name="0/90", upper_ply_index=0, lower_ply_index=0)
    assert not config_path.exists()

    specimen.save_config()
    saved = json.loads(config_path.read_text())
    assert [ply["name"] for ply in saved["plies"]] == ["0-degree"]
    assert [interface["name"] for interface in saved["interfaces"]] == ["0/90"]


def test_factory_saves_complete_definition_only_when_requested(tmp_path):
    specimen = Specimen.from_cross_ply(
        name="cross-ply",
        scale_px_mm=10.0,
        path_full="unused",
        sorting_key="frame",
        image_types=[".png"],
        auto_init_stacks=False,
        results_root=str(tmp_path / "results"),
    )

    assert not specimen.config_path().exists()
    specimen.save_config()

    saved = json.loads(specimen.config_path().read_text())
    assert [ply["orientation_deg"] for ply in saved["plies"]] == [0.0, 90.0]


@pytest.mark.parametrize("part", ["..", "nested/child", "/outside"])
def test_results_dir_rejects_path_traversal(tmp_path, part):
    specimen = make_specimen(tmp_path)

    with pytest.raises(ValueError, match="relative path component"):
        specimen.results_dir(part)


def test_specimen_name_must_be_a_single_path_component(tmp_path):
    with pytest.raises(ValueError, match="specimen name"):
        make_specimen(tmp_path, name="../outside")


def test_crack_subdirectory_cannot_escape_results_root(tmp_path):
    from deladect.io.cracks import crack_results_subdir

    specimen = make_specimen(tmp_path)
    ply = Ply("a/b", 0.0, 5.0, 10.0)
    target = crack_results_subdir(specimen, ply, "data")

    assert target.is_relative_to(specimen.results_root_path())
    with pytest.raises(ValueError, match="relative path component"):
        crack_results_subdir(specimen, ply, "../outside")
