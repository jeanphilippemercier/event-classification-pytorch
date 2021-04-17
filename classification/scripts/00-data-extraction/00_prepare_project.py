from uquake.core import read_inventory
from uquake.core.project_manager import ProjectManager
from uquake.core.settings import settings
from uquake.grid.nlloc import (ModelLayer, LayeredVelocityModel,
                               VelocityGridEnsemble)
from pathlib import Path
import numpy as np


test_artifact_path = Path(settings.TEST_ARTIFACTS)
inventory = read_inventory(str(test_artifact_path / 'inventory.xml'))


def make_layered_model():

    # The origin is the lower left corner
    project_code = settings.project.name
    network_code = settings.project.network
    origin = np.array(settings.grids.origin)
    dimensions = np.array(settings.grids.dimensions)
    spacing = np.array(settings.grids.spacing)

    z = [1168, 459, -300, -500]
    vp_z = [4533, 5337, 5836, 5836]
    vs_z = [2306, 2885, 3524, 3524]

    p_layered_model = LayeredVelocityModel(project_code)
    s_layered_model = LayeredVelocityModel(project_code, phase='S')
    for (z_, vp, vs) in zip(z, vp_z, vs_z):
        layer = ModelLayer(z_, vp)
        p_layered_model.add_layer(layer)
        layer = ModelLayer(z_, vs)
        s_layered_model.add_layer(layer)

    vp_grid_3d = p_layered_model.gen_3d_grid(network_code, dimensions, origin,
                                             spacing)
    vs_grid_3d = s_layered_model.gen_3d_grid(network_code, dimensions, origin,
                                             spacing)
    return VelocityGridEnsemble(vp_grid_3d, vs_grid_3d)


pm = ProjectManager(settings.project.base_directory,
                    settings.project.name,
                    settings.project.network)

velocities = make_layered_model()

pm.add_inventory(inventory, initialize_travel_time=False)
pm.add_velocities(velocities, initialize_travel_time=True)

