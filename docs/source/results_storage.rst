# How to store results

## Overview

DelaDect stores the results for each specimen in a separate folder:

```text
results/<specimen.name>/
```

By default, `results/` is a relative path, but it can be changed using `specimen.results_root`.

The `results_dir(*parts)` method can be used to create folders inside the specimen results directory. Each argument must be a single valid folder or file name, not a complete path.

## Configuration

The specimen configuration is saved in:

```text
results/<name>/config/<name>_config.json
```

Use `specimen.save_config()` to save the configuration. The path can also be obtained with `specimen.config_path()`, without saving anything.

Changes to the specimen, such as adding plies with `add_ply` or interfaces with `add_interface`, are only stored after saving the configuration again.

Alternatively, `save_specimen(specimen, path)` can be used to save the configuration to a specific location, as shown in the [multi-interface delamination example](examples/delamination_multi_interface.md).

## Crack detection

Crack detection results are organized by ply:

```text
results/<name>/cracks/ply_<sanitized_ply_name>/
├── data/
│   └── <specimen>_<sanitized_ply_name>_cracks.npz
├── metrics/
│   ├── rho_data.csv
│   └── crack_spacing.csv
└── plots/
    └── cracks_<label>.png
```

The `.npz` file contains the detected cracks for each frame, stored under keys such as `frame_0000`, `frame_0001`, and so on. The path to this file is also saved in `ply.metadata["crack_results_path"]`, so the results can be found again without having to reconstruct the folder path.

The `metrics` folder contains the crack density and spacing data, while `plots` contains the crack overlays for the individual frames.

## Delamination detection

The delamination masks are stored according to the type of analysis:

```text
results/<name>/<overlay_dirname>/<edge|diffuse|both|total>/
├── <masks_dirname>/
│   ├── edge_raw.npz
│   ├── edge_exclusion.npz
│   ├── diffuse_raw.npz
│   ├── diffuse_final.npz
│   └── combined.npz
└── overlays/
    ├── edge_overlay_<idx:04d>.png
    ├── diffuse_overlay_<idx:04d>.png
    ├── combined_overlay_<idx:04d>.png
    └── total_overlay_<idx:04d>.png

results/<name>/<overlay_dirname>/both/metrics/
└── frame_metrics.csv
```

By default, `overlay_dirname="delamination"` and `masks_dirname="masks"`. As with crack detection, each `.npz` file stores the masks per frame using the `frame_<idx:04d>` convention.

The paths to these files are also stored in `interface.metadata` under the following keys: `primary_masks_path`, `secondary_masks_path`, `diffuse_raw_masks_path`, `diffuse_masks_path`, `combined_masks_path`, and `delamination_metrics_path`.

For multi-interface edge delamination, the results follow a similar structure under `edge_multi/masks/`. Each interface has an `<interface>_inclusive.npz` and an `<interface>_exclusive.npz` file. More details are available in the [methodology](methodology.md).

## Preprocessing cache

Preprocessed images can also be cached to avoid repeating the same operations:

```text
results/<name>/<cache_dirname>/<key>/preprocess_%04d.npz
```

The default directory is `cache_dirname="Preprocessor_cache"`. The contents of each cached frame and the definition of `key` are explained in [Image preprocessing](Image_pre_processing.md).

## Saving and reloading a specimen

The specimen configuration and its stored results can be loaded again without repeating the crack and delamination detection. This is particularly useful when returning to a previously analysed specimen.

First, save the specimen manifest at the desired location:

```python
from deladect.io import save_specimen

manifest = specimen.results_dir("config") / "specimen.json"
save_specimen(specimen, manifest)
```

The same manifest can then be used to reload the specimen, either in the same session or in a new one:

```python
from deladect.io import load_specimen, load_stored_results

specimen = load_specimen(
    manifest,
    auto_init_stacks=False,
    load_results=True,
    verbose=True,
    strict=True,
)
bundles = load_stored_results(specimen, strict=True, verbose=True)
```

Setting `strict=True` raises an error if a file referenced in the metadata cannot be found, instead of silently skipping it.

If `strain_csv` was defined when creating the specimen, its path is also saved in the manifest. When the specimen is reloaded, `specimen.experimental_data` is populated automatically, without any additional steps.
