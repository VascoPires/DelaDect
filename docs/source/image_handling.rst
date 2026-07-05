Image Handling
==============

Overview
--------
DelaDect uses CrackDect image-stack backends to manage frame sequences.
It supports two backend modes:

- ``ImageStack`` (in-memory)
- ``ImageStackSQL`` (database-backed, memory-efficient)

For most workflows you interact with stacks through :class:`deladect.specimen.Specimen`.

Backend selection
-----------------
Selection is controlled by ``Specimen(stack_backend=...)``:

- ``"auto"`` (default): estimate stack memory and choose backend automatically
- ``"memory"``: force in-memory ``ImageStack``
- ``"sql"``: force ``ImageStackSQL``

Automatic mode compares estimated footprint to ``stack_limit_mb``.

Specimen example
----------------

.. code-block:: python

    from deladect.specimen import Specimen

    specimen = Specimen(
        name="Specimen A",
        scale_px_mm=40.0,
        path_full="path/to/full/images",
        path_upper_border="path/to/upper",
        path_lower_border="path/to/lower",
        path_middle="path/to/middle",
        sorting_key="_sc",
        image_types=[".png", ".jpg"],
        stack_backend="auto",      # auto, memory, or sql
        stack_limit_mb=512.0,
        sql_stack_kwargs={"database": "mydb", "stack_name": "specimen_stack"},
    )

    print(type(specimen.image_stack_full))

Manual frame assignment (no crackdect stack init)
--------------------------------------------------
For tests and notebooks where you want to bypass automatic stack initialization:

.. code-block:: python

    from pathlib import Path
    from skimage.io import imread

    specimen = Specimen(
        name="manual-stack",
        scale_px_mm=41.0,
        path_full="specimen_examples/sample-1",
        sorting_key="_sc",
        image_types=["png"],
        auto_init_stacks=False,
    )

    frame_paths = sorted(Path(specimen.path_full).glob("*.png"))
    specimen.path_full_list = [str(path) for path in frame_paths]
    specimen.image_stack_full = [imread(str(path)) for path in frame_paths]

Direct backend usage (advanced)
-------------------------------

.. code-block:: python

    import crackdect as cd
    import numpy as np

    mem_stack = cd.ImageStack(dtype=np.float32)
    mem_stack.add_image(np.zeros((256, 256), dtype=np.float32))

    sql_stack = cd.ImageStackSQL.from_paths(
        ["img1.png", "img2.png"],
        database="test",
        stack_name="stack1",
    )
    sql_stack.save_state()

Further reading
---------------
For backend internals and advanced operations, see CrackDect documentation:
`ImageStack intro <https://crackdect.readthedocs.io/en/latest/imagestack_intro.html>`_.
