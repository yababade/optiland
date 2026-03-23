Quickstart
==========

.. _first_script:

Once you have installed Optiland, you can start designing and analyzing optical systems. Here is a simple example that loads a Cooke Triplet lens system and visualizes it immediately in 2D.

What is updated in this guide
-----------------------------

This quickstart now emphasizes the fastest way to inspect a system from Python:

* ``draw()`` for a 2D optical layout with traced rays.
* ``draw3D()`` for a full 3D visualization when you want spatial context.
* A short reference for the most important ``draw()`` options so new users can adjust the first plot without searching elsewhere.

Optiland "Hello, World"
-----------------------

.. code-block:: python

   from optiland.samples.objectives import CookeTriplet

   lens = CookeTriplet()
   lens.draw()

.. figure:: images/cooke.png
   :alt: Cooke Triplet Lens System
   :align: center

   The Cooke triplet can be inspected immediately with ``draw()`` in 2D, and with ``draw3D()`` when a 3D view is needed.

Understanding ``draw()``
------------------------

``draw()`` is the standard 2D visualization entry point for an ``Optic``. It renders the optical layout, traces sample rays, and returns the Matplotlib ``Figure`` and ``Axes`` so you can continue customizing the plot in scripts or notebooks.

Common options include:

* ``num_rays``: controls how many rays are traced for each field and wavelength.
* ``projection``: chooses the 2D plane, such as ``"YZ"`` for the usual lens cross-section or ``"XY"`` for an aperture-style view.
* ``show_apertures``: overlays aperture graphics on the system view.
* ``title``, ``xlim``, and ``ylim``: help format the final plot for reports or notebooks.

For example:

.. code-block:: python

   fig, ax = lens.draw(
       num_rays=5,
       projection="YZ",
       show_apertures=True,
       title="Cooke Triplet Layout",
   )

If you need a volumetric view instead of a section view, use ``draw3D()``.

Running the GUI
---------------

Optiland includes a Graphical User Interface (GUI) for interactive design and analysis. Once the package is installed, you can launch the application from any terminal or console on your system by simply running the command:

.. code-block:: bash

   optiland

This will start the main application window. For development or troubleshooting, you can also run the GUI module directly using Python's ``-m`` flag:

.. code-block:: bash

   python -m optiland_gui.run_gui

For a more detailed guide on using the GUI, including an overview of its components and basic operations, please see the :ref:`gui_quickstart`.

Optiland for Beginners
----------------------

This script is the first of the learning guide series. It introduces the basic concepts of Optiland and demonstrates how to create a simple lens system.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>
