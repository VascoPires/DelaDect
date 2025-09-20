Image Handling
==============

DelaDect builds on the same **ImageStack** functionalities as CrackDect, but extends them and uses them exclusively as its backend for managing image data.

Available Backends
------------------

Two main classes are available for handling image stacks:

- **ImageStack**  
  A simple container that keeps all images in memory (RAM) at once.  
  Best suited for smaller datasets since it has very low overhead.

- **ImageStackSQL**  
  A database-backed container that only loads images into memory when needed.  
  This makes it more RAM-efficient and allows handling very large datasets.  
  It uses `sqlalchemy` to manage transactions, and can also persist image stacks for later use.

Automatic Selection
-------------------

When initializing, DelaDect automatically estimates the memory requirements of the image stack:

- If the estimated memory is below the configured threshold, **ImageStack** is used.
- If it exceeds the threshold, **ImageStackSQL** is used instead.

Manual Override
---------------

Users can manually configure the backend via the ``stack_backend`` setting:

- ``"auto"`` (default): Let DelaDect decide based on memory estimation.
- ``"memory"``: Always use **ImageStack**.
- ``"sql"``: Always use **ImageStackSQL**.

This manual setting overrides the automatic selection logic.

Specimen Example
----------------

The following shows how the backend is chosen when creating a :class:`Specimen`:

.. code-block:: python

    specimen = Specimen(
        name="Specimen A",
        dimensions={"width": 20, "thickness": 5},
        scale_px_mm=40.0,
        path_cut="path/to/cut/images",
        path_upper_border="path/to/upper",
        path_lower_border="path/to/lower",
        path_middle="path/to/middle",
        sorting_key="_sc",
        image_types=[".png", ".jpg"],
        stack_backend="auto",      # "auto", "memory", or "sql"
        stack_limit_mb=512.0,      # threshold in MB for auto mode
        sql_stack_kwargs={"database": "mydb", "stack_name": "specimen_stack"}
    )

    # Access the chosen backend for the cut region
    print(type(specimen.image_stack_cut))
    # -> <class 'ImageStack'> or <class 'ImageStackSQL'> depending on settings

Examples
--------
Since ImageStack is implemented as a backend for DelaDect, direct usage is not 
necessary. However, here are some basic examples of how to use both backends directly 
which can be used as for example indivudal inputs to some DelaDect functions.

Basic ImageStack
~~~~~~~~~~~~~~~~

.. code-block:: python

    import crackdect as cd
    stack = cd.ImageStack(dtype=np.float32)

    # Add images
    stack.add_image(img)

    # Access single image
    first = stack[0]

    # Access subset of images
    subset = stack[1:4]

    # Remove images
    del stack[2]

ImageStackSQL
~~~~~~~~~~~~~

.. code-block:: python

    import crackdect as cd
    stack = cd.ImageStackSQL(database="test", stack_name="stack1")

    # Add images from file paths
    stack = cd.ImageStackSQL.from_paths(["img1.png", "img2.png"], "test", "stack1")

    # Save current state
    stack.save_state()

    # Copy stack into a new table
    new_stack = stack.copy(stack_name="stack2")

For more advanced usage and details, please refer to the `CrackDect documentation <https://crackdect.readthedocs.io/en/latest/imagestack_intro.html>` .
