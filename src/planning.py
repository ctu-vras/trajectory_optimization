from queue import PriorityQueue
from enum import Enum
from queue import Queue
import numpy as np


class Action(Enum):
    """
    An action is represented by a 3 element tuple.
    
    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    LEFT = (0, -1, 1)
    RIGHT = (0, 1, 1)
    UP = (-1, 0, 1)
    DOWN = (1, 0, 1)
    LEFT_UP = (-1, -1, 1)
    LEFT_DOWN = (1, -1, 1)
    RIGHT_UP = (-1, 1, 1)
    RIGHT_DOWN = (1, 1, 1)
    
    @property
    def cost(self):
        return self.value[2]
    
    @property
    def delta(self):
        return (self.value[0], self.value[1])
            
    
def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node
    
    # check if the node is off the grid or
    # it's an obstacle
    
    if x - 1 < 0 or grid[x-1, y] == 1:
        valid.remove(Action.UP)
    if x + 1 > n or grid[x+1, y] == 1:
        valid.remove(Action.DOWN)
    if y - 1 < 0 or grid[x, y-1] == 1:
        valid.remove(Action.LEFT)
    if y + 1 > m or grid[x, y+1] == 1:
        valid.remove(Action.RIGHT)
        
    if (x - 1 < 0 or y - 1 < 0) or grid[x - 1, y - 1] == 1:
        valid.remove(Action.LEFT_UP)
    if (x - 1 < 0 or y + 1 > m) or grid[x - 1, y + 1] == 1:
        valid.remove(Action.RIGHT_UP)
    if (x + 1 > n or y - 1 < 0) or grid[x + 1, y - 1] == 1:
        valid.remove(Action.LEFT_DOWN)
    if (x + 1 > n or y + 1 > m) or grid[x + 1, y + 1] == 1:
        valid.remove(Action.RIGHT_DOWN)
        
    return valid

def heuristic(p, goal, mode='euclid'):
    p = np.array(p); goal = np.array(goal)
    if mode=='euclid':
        h = np.sqrt( (p[0] - goal[0])**2 + (p[1]-goal[1])**2 )
    elif mode=='manhattan':
        h = np.abs(p[0] - goal[0]) + np.abs(p[1]-goal[1])
    else:
        h = np.linalg.norm(p - goal)
    return h

def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    while not queue.empty():
        item = queue.get()
        current_node = item[1]

        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost



def breadth_first_search(grid, start, unexplored_value=0.5):
    q = Queue(); q.put(start)
    visited = set(); visited.add(start)
    branch = {}
    found = False
    goal = None
    
    # Run loop while queue is not empty
    while not q.empty():
        # get element from the queue
        current_node = q.get()
        # if exploration is reached a frontier
        if grid[current_node[0], current_node[1]] == unexplored_value:
            print('BFS: found a frontier.')
            goal = current_node
            found = True
            break
        else:
            # Iterate through each of the new nodes and:
            # If the node has not been visited you will need to
            # 1. Mark it as visited
            # 2. Add it to the queue
            # 3. Add how you got there to the branch dictionary
            valid = valid_actions(grid, current_node)
            for action in valid:
                # delta of performing the action
                da = action.value
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                # Check if the new node has been visited before.
                # If the node has not been visited you will need to
                # 1. Mark it as visited
                # 2. Add it to the queue
                # 3. Add how you got there to branch
                if next_node not in visited:                
                    visited.add(next_node)               
                    q.put(next_node)          
                    branch[next_node] = (current_node, action) 
    # Now, if you found a path, retrace your steps through 
    # the branch dictionary to find out how you got there!
    path = []
    if found:
        # retrace steps
        path = []
        n = goal
        while branch[n][0] != start:
            path.append(branch[n][0])
            n = branch[n][0]
        path.append(branch[n][0])
            
    return path[::-1], goal


def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon
# using collinearity check here
def prune_path(path, eps=1e-6):
    pruned_path = path  
    i = 0
    while i < len(pruned_path) - 2:
        p1 = point(pruned_path[i])
        p2 = point(pruned_path[i+1])
        p3 = point(pruned_path[i+2])
        # If the 3 points are in a line remove
        # the 2nd point.
        # The 3rd point now becomes and 2nd point
        # and the check is redone with a new third point
        # on the next iteration.
        if collinearity_check(p1, p2, p3, eps):
            # Something subtle here but we can mutate
            # `pruned_path` freely because the length
            # of the list is check on every iteration.
            pruned_path.remove(pruned_path[i+1])
        else:
            i += 1
    return np.array(pruned_path)

from scipy import interpolate
def smooth_path(path, vis=False):
    x, y = zip(*path[:,:2])
    #create spline function
    f, u = interpolate.splprep([x, y], s=0)
    #create interpolated lists of points
    xint, yint = interpolate.splev(np.linspace(0, 1, 50), f)
    if vis:
        plt.scatter(x, y)
        plt.plot(xint, yint)
    return np.vstack([xint, yint]).T

# APF
from scipy.ndimage.morphology import distance_transform_edt as bwdist

def construct_path(total_potential, start_coords, end_coords, max_its):
    # construct_path: This function plans a path through a 2D
    # environment from a start to a destination based on the gradient of the
    # function f which is passed in as a 2D array. The two arguments
    # start_coords and end_coords denote the coordinates of the start and end
    # positions respectively in the array while max_its indicates an upper
    # bound on the number of iterations that the system can use before giving
    # up.
    # The output, route, is an array with 2 columns and n rows where the rows
    # correspond to the coordinates of the robot as it moves along the route.
    # The first column corresponds to the x coordinate and the second to the y coordinate
    gy, gx = np.gradient(-total_potential)
    route = [np.array(start_coords)]
    for i in range(max_its):
        current_point = np.array(route[-1])
        #print(sum( abs(current_point-end_coords) ))
        if sum( abs(current_point-end_coords) ) < 2.0:
            print('Reached the goal !')
            break
        ix = np.clip(int(current_point[1]), 0, gx.shape[0]-1)
        iy = np.clip(int(current_point[0]), 0, gx.shape[1]-1)
        vx = gx[ix, iy]; vy = gy[ix, iy]
        dt = 0.5/(1e-8+np.linalg.norm([vx, vy]))
        next_point = current_point + dt*np.array( [vx, vy] )
        route.append(next_point)
    return route

def apf_planner(grid, start, goal, num_iters=100, influence_r=0.2, repulsive_coef=200, attractive_coef=0.01):
    nrows, ncols = grid.shape
    x, y = np.meshgrid(np.arange(ncols), np.arange(nrows))
    # Compute repulsive force
    d = bwdist(grid==0)
    # Rescale and transform distances
    d2 = d/100. + 1
    d0 = 1 + influence_r
    nu = repulsive_coef
    repulsive = nu*((1./d2 - 1./d0)**2)
    repulsive[d2 > d0] = 0

    # Compute attractive force
    xi = attractive_coef
    attractive = xi * ( (x - goal[0])**2 + (y - goal[1])**2 )
    # Combine terms
    total = attractive + repulsive
    # plan a path
    path = construct_path(total, start, goal, num_iters)
    return path

def apf_path_to_map(apf_path, elev_map, elev_grid, map_res=0.15):
    path_map = []
    x_min, y_min = np.min(elev_map[:, 0]), np.min(elev_map[:, 1])
    for point in apf_path:
        z = elev_grid[int(point[1]), int(point[0])]
        p = (np.array(point)*map_res+[y_min, x_min]).tolist() + [z]
        path_map.append([p[1], p[0], p[2]])
    return path_map

# def apf_planner(grid, start, goal, elev_map, elev_grid):
#     path_grid = apf_grid_planner(grid, start, goal)
#     path_map = apf_path_to_map(apf_path, elev_map, elev_grid)
#     return path_map