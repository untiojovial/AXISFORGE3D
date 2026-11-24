# UR5 sample

`robot.urdf` and `meshes/*.stl` are copied unmodified from the
`example-robot-data` project (`robots/ur_description/urdf/ur5_robot.urdf` and
`robots/ur_description/meshes/ur5/collision/*.stl`, branch `master`), under the
BSD 3-Clause License in `LICENSE` in this folder. The viewer build embeds them
so the page can demonstrate URDF import without the user supplying files, and
reproduces the license in the page.

Only the collision meshes are included; the visual meshes are COLLADA, which
the importer does not read, so the collision meshes are drawn instead.
