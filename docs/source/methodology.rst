Delamination Detection Methodology
==================================

Edge and diffuse delamination are both reached through a single
:class:`~deladect.detection.delamination.DelaminationDetector` per
``(specimen, interface)`` pair. Then two other detectors are available inside
the same object, for the two delamination modes:
``detector.edge`` and
``detector.diffuse``.

Detection modes
---------------

DelaDect distinguishes two delamination modes, each documented on its own page:

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Edge delamination
      :link: edge_delamination
      :link-type: doc

      Damage connected to a specimen free edge: directional reconstruction
      and frame-to-frame latching.

   .. grid-item-card:: Diffuse delamination
      :link: diffuse_delamination
      :link-type: doc

      Damage sought locally around tracked transverse cracks, using a
      per-crack baseline to isolate new darkening.

   .. grid-item-card:: Multi-interface delamination
      :link: multi_interface_delamination
      :link-type: doc

      Attributing later damage to the correct, deeper interface in
      laminates with more than two plies. Edge-only.

.. toctree::
   :maxdepth: 1
   :hidden:

   edge_delamination
   diffuse_delamination
   multi_interface_delamination

Combining the two
------------------

:meth:`~deladect.detection.delamination.DelaminationDetector.detect_both_delaminations`
runs both pipelines together and resolves any overlap
between the two modes favouring the edge delamination.


See also
--------

- :doc:`detection` for the callable API and default parameter values.
