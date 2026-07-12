Image Preprocessing
===================

Overview
--------
Preprocessing is the normalization stage used before crack and delamination
detection. It is especially important for edge and multi-interface delamination,
where temporal consistency strongly depends on reference-frame strategy.

Why this step matters
---------------------
- Edge delamination assumes the specimen edges remain aligned across frames.
- Diffuse and combined workflows rely on normalized intensity ranges.
- Multi-interface promotion is sensitive to reference adaptation speed.

If image drift is present, run the shift-correction workflow first.

Core API
--------
Use :meth:`deladect.detection.delamination.DelaminationDetector.preprocess_stack_to_disk`
to persist processed frames and reuse them in later detection runs.

.. code-block:: python

    detector = DelaminationDetector(specimen, interface, save_preprocess_outputs=True)

    cache_paths = detector.preprocess_stack_to_disk(
        specimen.image_stack_full,
        key="sample_preprocess",
        history_mode="running",
        reference_mode="rolling_median",
        reference_window=7,
        reference_skip=2,
        cache_dirname="Preprocessor_cache",
    )["cache_paths"]

Reference strategy guidance
---------------------------
- ``reference_mode="static"``: stable and reproducible baseline from one fixed
  frame (index selected by ``reference_skip``). Good default for initial
  single-interface checks.
- ``reference_mode="rolling_median"``: adapts to gradual illumination/background
  drift. Recommended for ``detect_edge_multi`` with ``reference_skip >= 1``.
- ``reference_skip``: number of newest previous frames excluded from the rolling
  reference. Larger values reduce self-cancellation of recently formed damage.

Single-frame rolling reference (legacy rolling-style behavior)
----------------------------------------------------------------
If you want a rolling reference based on one prior frame, use:

.. code-block:: python

    reference_mode="rolling_median"
    reference_window=1

Then ``reference_skip`` controls how far behind the reference sits:

- ``reference_skip=0`` -> approximately ``n-1``
- ``reference_skip=1`` -> approximately ``n-2``
- ``reference_skip=2`` -> approximately ``n-3``

Early frames with insufficient history fall back to the current frame.

Static vs rolling references in highly damaged stages
-----------------------------------------------------
In highly damaged specimens, a static baseline can lose sensitivity because most
pixels already differ strongly from the initial frame. When this happens, newly
forming damage is harder to separate from previously damaged regions.

Rolling references are usually better in this regime. Common schemes are:

1. frame-to-frame reference (``n-1``): highlights rapid changes but may suppress
   slowly growing delamination and can accumulate noise.
2. rolling-median reference (median of previous ``m`` frames): robust compromise
   for long sequences; keeps the baseline behind the current frame while
   resisting outliers and illumination spikes.
3. rolling static-segment reference (lagged frame ``n-m``): fixed temporal lag
   without median smoothing.

Practical recommendation:

- use ``reference_mode="static"`` in early/clean stages,
- switch to rolling schemes as damage density increases,
- keep the reference "behind" the current frame (for example via
  ``reference_skip``) so slow-growing damage is not immediately normalized away.

Quick decision guide
--------------------
- If edge and diffuse masks are both noisy: verify shift correction first.
- If diffuse appears over-detected: increase ``reference_skip`` and reduce
  post-threshold closing before changing many filter parameters.
- If late-stage damage is under-emphasized: try ``rolling_median`` with a larger
  ``reference_window``.
- If early clean frames are over-normalized: prefer ``static`` baseline for the
  first benchmark pass.

The public preprocessing API currently exposes ``static`` and
``rolling_median`` directly. Frame-to-frame and lagged-reference behavior can be
approximated in notebook workflows by tuning ``reference_window`` and
``reference_skip``.

History clamp guidance
----------------------
The optional minimum-history stage (running or rolling) can suppress bright flicker
before reference normalization.

- ``history_mode="running"``: minimum over all previous frames.
- ``history_mode="rolling"``: minimum over a sliding window
  (configure with ``history_window_size``).

Outputs
-------
Preprocessing writes two output families under the specimen result root:

- cache bundles: ``Preprocessor_cache/<key>/preprocess_XXXX.npz``
  plus ``Preprocessor_cache/<key>/preprocess_manifest.json``
- optional previews: ``Preprocessor_outputs/<key>/``
  (raw, baseline, processed triplets)

Reusing cached preprocessing
----------------------------
Most detection methods accept preprocessed cache paths. Reusing these paths keeps
parameter studies reproducible and avoids re-running normalization.

.. code-block:: python

    edge_masks = detector.edge.detect_primary(
        processed_cache_paths=cache_paths,
        save_overlays=False,
    )["masks"]

    diffuse_masks = detector.diffuse_delamination(
        cracks=cracks,
        processed_cache_paths=cache_paths,
        save_overlays=False,
    )["masks"]

Practical checks
----------------
- Verify the specimen remains horizontally aligned at top and bottom edges.
- Inspect processed previews for clipping or over-normalization.
- Keep preprocessing settings fixed when comparing thresholds across specimens.

Common pitfalls
---------------
- Using rolling references with no skip can dampen recent changes too aggressively.
- Mixing preprocessed caches generated with different settings in one comparison.
- Running edge workflows on unaligned stacks (false edge activations).
