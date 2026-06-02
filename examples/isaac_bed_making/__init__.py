"""Two Unitree G1 humanoids autonomously make a bed in NVIDIA Isaac Sim,
coordinating as equal peers over Arm Device Connect.

This is a self-contained example (it does not modify the ``strands_robots``
product package or Arm's Device Connect). It runs under the Isaac Sim / Isaac
Lab Python on a DGX Spark and reuses the Device Connect swarm driver from
``examples/unitree_g1_bed_making_g1_driver.py``.

Modules:

* :mod:`cloth`        — PhysX particle-cloth bedsheet + grasp attachment.
* :mod:`coordination` — in-process Device Connect swarm of two G1 peers.
* :mod:`behavior`     — per-robot autonomous state machine (non-scripted).
* :mod:`scene`        — build the room/bed/sheet/robots in Isaac Sim.
* ``demo.py``         — entrypoint: ``isaaclab.sh -p examples/isaac_bed_making/demo.py``.
"""
