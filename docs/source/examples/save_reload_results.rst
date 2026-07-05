Save and Reload Results
=======================

This example shows how to persist a specimen manifest and later reload the
specimen together with stored crack and delamination artefacts.

Save workflow state
-------------------

.. code-block:: python

    from deladect.io import save_specimen

    manifest = specimen.results_dir("config") / "sample-1-save-reload.json"
    save_specimen(specimen, manifest)

Reload and discover stored artefacts
------------------------------------

.. code-block:: python

    from deladect.io import load_specimen, load_stored_results

    specimen_new = load_specimen(
        manifest,
        auto_init_stacks=False,
        load_results=True,
        verbose=True,
    )

    bundles = load_stored_results(specimen_new, strict=False, verbose=False)
    print(bundles.keys())

Expected console messages
-------------------------
When ``verbose=True`` in ``load_specimen(..., load_results=True)``, messages are
printed for discovered artefacts, for example:

- ``Found cracks for ply 'ply_0' (...)``
- ``Found edge/diffuse delamination artefacts for interface 'i0': ...``

Strict mode
-----------
Use ``strict=True`` to fail fast when a metadata path points to a missing file.

.. code-block:: python

    specimen_new = load_specimen(
        manifest,
        auto_init_stacks=False,
        load_results=True,
        verbose=True,
        strict=True,
    )
