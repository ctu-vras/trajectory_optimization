import numpy as np


def create_grid(elev_map, robot_z, map_res=0.15, safety_distance=0.15, margin=0.3, unexplored_value=0.5):
    # minimum and maximum east coordinates
    x_min = np.min(elev_map[:, 0])
    x_max = np.max(elev_map[:, 0])
    
    # minimum and maximum north coordinates
    y_min = np.min(elev_map[:, 1])
    y_max = np.max(elev_map[:, 1])

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    y_size = int((y_max - y_min)//map_res)
    x_size = int((x_max - x_min)//map_res)

    # Initialize an unexplored grid
    # unexplored: 0.5
    # free:        0
    # occupied:    1
    grid = np.zeros((x_size, y_size)) + unexplored_value
    elev_grid  = np.full((x_size, y_size), np.nan)

    # Populate the grid with obstacles
    for i in range(elev_map.shape[0]):
        x, y, z, _ = elev_map[i, :]
        dx, dy, dz = map_res, map_res, z-robot_z
        sd = safety_distance * (z > margin) # safety distance is added to points treated as obstacles
        obstacle = [
            int(np.clip((x - dx - sd - x_min)//dx, 0, x_size-1)),
            int(np.clip((x + dx + sd - x_min)//dx, 0, x_size-1)),
            int(np.clip((y - dy - sd - y_min)//dy, 0, y_size-1)),
            int(np.clip((y + dy + sd - y_min)//dy, 0, y_size-1)),
        ]
        # If cell height is higher than robot location more than heght margin
        # than it is not traversable, i.e. treated as obstacle.
        # Otherwise the cell is free to go to.
        grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = dz > margin
        elev_grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = z
    return grid, elev_grid