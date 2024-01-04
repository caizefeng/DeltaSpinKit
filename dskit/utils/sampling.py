import numpy as np


def spherical_uniform(grid, end_point):
    thetas = np.linspace(0.0, np.pi, grid[0], endpoint=end_point)
    phis = np.linspace(0.0, 2 * np.pi, grid[1], endpoint=False)
    system_scalar_matrix = np.zeros((thetas.shape[0], phis.shape[0]))
    return thetas, phis, system_scalar_matrix
