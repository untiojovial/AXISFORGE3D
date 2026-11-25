"""Three fixed SI geometries and the original cube, using the shared solver."""

from research_grasping.simulation import mujoco_pilot

# Each identifier selects the entire geometry; callers cannot mix dimensions or units.
GEOMETRIES = {
    "cube": ("box", ".02 .02 .02"),
    "elongated_box": ("box", ".02 .03 .02"),
    "vertical_cylinder": ("cylinder", ".02 .02"),
    "ellipsoid": ("ellipsoid", ".02 .03 .02"),
}


def evaluate(
    object_id="cube",
    offset_x_m=0.0,
    mass_kg=0.05,
    friction=1.0,
    timestep_s=0.001,
    solver="Newton",
):
    """Evaluate an exact manifest geometry in the shared MuJoCo scene."""
    if object_id not in GEOMETRIES:
        raise ValueError("unknown primitive geometry")
    shape, size = GEOMETRIES[object_id]
    return mujoco_pilot._evaluate(
        shape, size, offset_x_m, mass_kg, friction, timestep_s, solver
    )
