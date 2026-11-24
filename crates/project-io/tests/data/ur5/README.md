# UR5 test model

`ur5_robot.urdf` is copied unmodified from the `example-robot-data` project
(`robots/ur_description/urdf/ur5_robot.urdf`, branch `master`), and so are
`visual/base.dae` and `visual/wrist3.dae`
(`robots/ur_description/meshes/ur5/visual/`). They are distributed under the
BSD 3-Clause License reproduced in `LICENSE` in this folder.

They are used only as test input. The URDF test does not need the meshes and
reports them as missing; the COLLADA test compares the two visual meshes with
the matching collision meshes in `apps/viewer/samples/ur5/meshes`, which were
modelled separately.
