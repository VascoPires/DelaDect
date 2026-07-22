Image Pre-processing
=====================

Overview
--------
Before any crack or delamination detector looks at a frame, DelaDect runs a
stack-level normalization pass: :meth:`~deladect.detection.delamination.DelaminationDetector.preprocess_stack_to_disk`.
This is separate from (and runs before) the per-slice filtering described in
:doc:`delamination`'s "Algorithm summary" (directional grey-opening, unsharp
mask, directional Gaussian smoothing) — this page covers the stack-wide
step that conditions raw frames, not a specific detector's own filter
chain.

Two mechanisms are applied per frame, in order:

1. **History clamp** — suppresses transient bright noise/reflections.
2. **Reference normalization** — a ratio against a baseline frame, so new
   damage stands out relative to that baseline instead of absolute
   brightness.

.. currentmodule:: deladect.detection.delamination

History clamp
--------------
``history_clamp=True`` (default) maintains a running or rolling-window
pixel-wise minimum across prior frames and clamps each frame to
``min(current_frame, history)``. Only pixel values that have "gone dark"
and *stayed* dark survive; frame-to-frame flicker (reflections, transient
bright artifacts) is suppressed, isolating persistent damage signal.

Controlled by ``history_mode`` (``"running"`` default, or a rolling window)
and ``history_window_size``.

Reference normalization
------------------------
The core operation is literal elementwise division against a baseline
frame (:func:`_normalize_reference_frame`, implementation detail):

.. code-block:: python

   denominator = np.maximum(baseline_float, 1e-3)
   ratio = np.clip(frame_float / denominator, 0.0, 1.0)
   processed = (ratio * 255.0).astype(np.uint8)

This is a flat-field-style illumination correction: it cancels out slow
lighting drift shared between a frame and its baseline, so what's left is
dominated by genuinely new, persistent darkening (damage) rather than
absolute brightness.

``reference_mode`` selects how the baseline is chosen:

- ``"static"`` (default) — one fixed early frame, reused for the whole
  stack. Standard for :meth:`EdgeDetector.detect_primary`,
  :meth:`DelaminationDetector.detect_both_delaminations`, and
  :meth:`DiffuseDetector.diffuse_delamination`.
- ``"rolling_median"`` — an adaptive baseline: the median of a trailing
  window of recent frames, ``[start_idx, end_idx)`` where
  ``end_idx = idx - reference_skip`` and ``start_idx = end_idx - reference_window``.
  Reserved for :meth:`EdgeDetector.detect_edge_multi`.

Static-reference limits and the rolling alternative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A static baseline is simple and works well when the damage front doesn't
overlap the baseline's own condition. It becomes a problem specifically
for **multi-interface promotion** (``detect_edge_multi``): if the baseline
frame is far in the past, damage already present in *both* the current
frame and the (also-damaged-by-then) baseline can partially cancel out in
the ratio, weakening the signal exactly where promotion needs to detect
it ("self-canceling reference behavior").

``reference_mode="rolling_median"`` avoids this by keeping the baseline
close to "now" — a lagged, adaptive reference instead of a single fixed
point in time. Recommended settings for ``detect_edge_multi``:

- ``reference_mode="rolling_median"``
- ``reference_skip >= 1``

With ``reference_window=1`` this behaves as a simple lagged single-frame
reference: ``reference_skip=0`` uses frame ``n-1`` as baseline,
``reference_skip=1`` uses ``n-2``, and so on. ``detect_edge_multi``'s
auto-preprocessing path defaults to ``reference_window=10``,
``reference_skip=1`` when neither is supplied.

API: ``preprocess_stack_to_disk``
----------------------------------
.. code-block:: python

   detector.preprocess_stack_to_disk(
       stack,
       key="edge_primary_auto_0/90",
       max_frames=None,
       history_mode="running",
       history_window_size=None,
       reference_mode="static",
       reference_window=10,
       reference_skip=0,
       cache_dirname="Preprocessor_cache",
       progress=False,
   )

Returns ``{"cache_paths": [...]}``, one path per processed frame. Each
cached ``.npz`` lives at ``<results>/<cache_dirname>/<key>/preprocess_%04d.npz``
and stores:

- ``processed`` — the normalized frame
- ``baseline`` — the baseline frame used for that index
- ``ref_start_idx``, ``ref_end_idx``, ``ref_anchor_idx`` — the resolved
  reference window bounds
- ``reference_mode``, ``reference_window``, ``reference_skip``,
  ``history_mode``, ``history_window_size`` — the settings used, so the
  cache is self-describing

A manifest file in the same cache directory records the run's settings for
reuse across detectors.

Choosing a mode in practice
-----------------------------
- Running :meth:`EdgeDetector.detect_primary`, diffuse detection, or
  combined edge+diffuse on a single interface: the ``reference_mode="static"``
  default is normally fine.
- Running :meth:`EdgeDetector.detect_edge_multi` (multi-interface
  promotion): use ``reference_mode="rolling_median"`` with
  ``reference_skip >= 1``, per the guidance above.
- If diffuse masks look too broad (rectangular ROI-like shapes), see the
  troubleshooting order in :doc:`delamination`'s "Diffuse troubleshooting"
  section — ``reference_skip`` is the first knob to try there too.


