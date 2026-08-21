03 - Specimen I/O: Save, Reload, and Experimental Data
=======================================================

This example shows how to persist a specimen manifest and later reload the
specimen together with its stored crack and delamination artefacts -- and
its experimental data, since ``strain_csv`` is part of the saved manifest.

Its goal is reproducibility rather than image generation. It makes sense as a
separate example because analysis and later inspection often happen in
different Python sessions.

Run ``python examples/01_getting_started.py`` first, followed by:

.. code-block:: bash

   python examples/03_save_reload_results.py

``01_getting_started.py`` builds its specimen from ``example_images/sample-1``
with ``strain_csv=".../sample-1/experimental_data.csv"``, then saves the
manifest via ``save_specimen``. This example only reloads:

.. code-block:: python

    from deladect.io import load_specimen, load_stored_results

    specimen = load_specimen(
        manifest,
        auto_init_stacks=False,
        load_results=True,
        verbose=True,
        strict=True,
    )
    bundles = load_stored_results(specimen, strict=True, verbose=True)

Experimental data on reload
----------------------------

``strain_csv`` round-trips through ``to_dict``/``from_dict`` like any other
constructor argument, so ``specimen.experimental_data`` is populated again as
soon as the specimen is reconstructed -- no separate reload step is needed:

.. code-block:: python

    print(specimen.experimental_data)

Expected console messages
--------------------------
When ``verbose=True`` in ``load_specimen(..., load_results=True)``, messages
are printed for discovered artefacts, for example:

- ``Found cracks for ply 'ply_0' (...)``
- ``Found edge/diffuse delamination artefacts for interface 'i0': ...``

Strict mode
-----------
Use ``strict=True`` to fail fast when a metadata path points to a missing
file, rather than silently skipping it.

For the on-disk layout these calls read from, see :doc:`../results_storage`.
