Image Handling
===============

Overview
--------
This page covers how raw image sequences get loaded into a
:class:`~deladect.specimen.Specimen` — folder conventions, sorting, and the
resulting in-memory structure. It's the step before any preprocessing or
detection runs; see :doc:`Image_pre_processing` for what happens to pixels
once they're loaded.

.. currentmodule:: deladect.specimen

Loading a region
-----------------
A ``Specimen`` is constructed with one or more region paths — ``path_full``
(required) and optionally ``path_upper_border``, ``path_lower_border``,
``path_middle``. Each region is loaded independently by
``_load_region_stack``:

1. Collect image paths in the folder via crackdect's ``image_paths(folder,
   image_types=...)``. ``image_types`` defaults to ``["png", "jpg", "bmp"]``;
   values are lowercased and any leading dot is stripped, so ``".PNG"`` and
   ``"png"`` are equivalent.
2. Sort them via crackdect's ``sort_paths(paths, sorting_key=self.sorting_key)``.
   ``sorting_key`` is the substring/pattern used to order frames
   correctly — get this wrong and frames load in filesystem order instead
   of acquisition order.
3. Build an in-memory (or on-disk-backed) image stack.

If ``auto_init_stacks=True`` (the default), this happens automatically in
``__post_init__`` right after construction — no separate "load" call is
needed, and nothing is written to disk at this stage.

In-memory structure
---------------------
After loading, a ``Specimen`` exposes, per region (``full``, ``upper``,
``lower``, ``middle``):

- ``path_<region>_list`` — the sorted list of file paths (``List[str]``).
- ``image_stack_<region>`` — the loaded stack, or ``None`` if that region
  wasn't provided.

The stack itself is either held fully in memory or backed by SQLite,
depending on ``stack_backend`` (``"auto"`` by default) and
``stack_limit_mb`` (default 2048.0) — large sequences that would exceed
the memory budget are automatically switched to the disk-backed form
without changing how you use the stack.

Only ``path_full`` is required; ``upper``/``lower``/``middle`` are optional
and simply stay unset (``image_stack_<region> = None``) if not given.

Folder naming: "cut" is a convention, not a requirement
----------------------------------------------------------
You'll see example folders named ``cut`` or ``cut_images`` throughout the
docs and sample data (e.g. ``Specimen.from_cross_ply(path_full="cut_dir", ...)``).
This means "raw acquisition frames trimmed to the specimen region" — it is
**not** a hardcoded name DelaDect looks for, and it does **not** mean
"already shift-corrected." Point ``path_full`` at whatever folder holds
your sequence; shift correction (see :doc:`shift_correction`) is a
separate step you run beforehand, producing its own
``shift_corrected/`` output folder that you then point subsequent
``Specimen`` instances at.

Building a specimen the easy way
-----------------------------------
``Specimen.from_cross_ply(name=..., scale_px_mm=..., path_full=...,
sorting_key=..., image_types=..., ...)`` wraps construction plus adding two
``[0, 90]`` plies and one ``"0/90"`` interface — the common cross-ply case.
See :doc:`examples/getting_started` for a full worked example.

Related pages
-------------
- :doc:`Image_pre_processing` for what happens to a loaded stack next.
- :doc:`shift_correction` for aligning raw frames before loading them here.
- :doc:`results_storage` for where output goes once detection has run.
