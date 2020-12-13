import math
import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull


def sphericalFlip(points, center, param=2):
    '''
    Function used to Perform Spherical Flip on the Original Point Cloud
    '''
    n = len(points) # total n points
    points = points - np.repeat(center, n, axis = 0) # Move C to the origin
    normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + z^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis=0) # Radius of Sphere

    flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
    flippedPoints += points 

    return flippedPoints

def convexHull(points):
    '''
    Function used to Obtain the Convex hull
    '''
    points = np.append(points, [[0,0,0]], axis = 0) # All points plus origin
    hull = ConvexHull(points) # Visible points plus possible origin. Use its vertices property.

    return hull

def hidden_pts_removal(pts_np, R_param=2):
    # Use a flag array indicating visibility
    flag = np.zeros(len(pts_np), int)
    # Initialize the points visible from camera location, C. 0 - Invisible; 1 - Visible.
    C = np.array([0, 0, 0])  # camera center

    flippedPoints = sphericalFlip(pts_np, C[None], R_param)
    myHull = convexHull(flippedPoints)
    visibleVertex = myHull.vertices[:-1] # indexes of visible points
    flag[visibleVertex] = 1
    #invisibleId = np.where(flag == 0)[0] # indexes of the invisible points

    pts_np_visible = pts_np[visibleVertex, :]
    return pts_np_visible
