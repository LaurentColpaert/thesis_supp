"""
Laurent Colpaert - Thesis 2022-2023
"""
import ast
from math import cos, sin, sqrt
from xml.dom import minidom



def retrieve_patches(argos_file : str, mission) -> tuple:
    from simulation import Mission
    """
    Retrieve the different values of the landmarks inside the argos file

    Args:
        -None
    Returns:
        -tuple: the circles or rectangles and the obstacles
    """
    patches = []
    goal = []

    # parse an xml file by name
    file = minidom.parse(f'/home/laurent/Documents/Polytech/MA2/thesis/examples/argos/{argos_file}')

    #retriving circle patches
    circles = file.getElementsByTagName('circle')
    for c in circles:
        if (mission == Mission.AAC or mission == Mission.HOMING) and c.getAttribute("color") != "white":
            goal = ast.literal_eval("[" + c.getAttribute("position") + "," + c.getAttribute("radius") + "]")
        patches.append(ast.literal_eval("[" + c.getAttribute("position") + "," + c.getAttribute("radius") + "]"))
    #retriving rect patches
    rectangles = file.getElementsByTagName('rectangle')
    for r in rectangles:
        if (r.getAttribute("color") == "white") and mission == mission.SHELTER:
            goal = ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]")
        patches.append(ast.literal_eval("[" + r.getAttribute("center") + "," + r.getAttribute("width") + "," + r.getAttribute("height") + "]"))
    obstacles = []
    boxes = file.getElementsByTagName('box')
    for b in  boxes:
        if("obstacle" in b.getAttribute("id")):
            body = b.getElementsByTagName("body")[0]
            center = ast.literal_eval("[" + body.getAttribute("position") + "]")[:-1]
            width = ast.literal_eval("[" + b.getAttribute("size") + "]")[1]
            orientation = ast.literal_eval("[" + body.getAttribute("orientation") + "]")[0]
            a = [center[0] + width*sin(orientation), center[1] + width*cos(orientation)]
            b = [center[0] - width*sin(orientation), center[1] - width*cos(orientation)]
            obstacles.append([a,b])

    return patches, obstacles, goal

def distToCircle(circle : tuple, pos : tuple, obstacles : list, arenaD : float)-> float:
    """
    Compute the distance of a point to a circle

    Args:
        -circle (tuple): (x,y,radius)
        -pos (tuple): (x,y) = the position of the point
    Returns:
        -float: the distance
    """
    c_x = circle[0]
    c_y = circle[1]
    r = circle[2]
    for obs in obstacles:
        if(intersect(pos,circle,obs[0], obs[1])):
            return arenaD
    return max(0, sqrt((pos[0]-c_x)**2 + (pos[1] - c_y)**2) - r)

def distToRect(rect : tuple, pos : tuple, obstacles : list, arenaD : float)-> float:
    """
    Compute the distance of a point to a rectangle

    Args:
        -rect (tuple): (x,y,width,length)
        -pos (tuple): (x,y) = the position of the point
    Returns:
        -float: the distance
    """
    x_min = rect[0] - rect[2]/2
    x_max = rect[0] + rect[2]/2
    y_min = rect[1] - rect[3]/2
    y_max = rect[1] + rect[3]/2

    dx = max(x_min - pos[0], 0, pos[0] - x_max)
    dy = max(y_min - pos[1], 0, pos[1] - y_max)
    
    for obs in obstacles:
        if(intersect(pos,[x_min,pos[1]],obs[0], obs[1]) or
            intersect(pos,[x_max,pos[1]],obs[0], obs[1]) or
            intersect(pos,[pos[0],y_min],obs[0], obs[1]) or
            intersect(pos,[pos[0],y_max],obs[0], obs[1])):
            return arenaD
    return sqrt(dx**2 + dy**2)

def ccw(a : float, b : float, c : float) -> bool:
    """
    Conter-clockwise function
    """
    return (c[0] - a[0])*(b[1] - a[1]) > (b[0] - a[0])*(c[1] - a[1])

def intersect( a : float, b : float, c : float, d : float)-> bool:
    """
    Return true if segments AB and CD intersect

    Args:
        -a (float): the x of the first position
        -b (float): the y of the first position
        -c (float): the x of the second position
        -d (float): the y of the second position
    Returns:
        -bool: intersect or not
    """
    return (ccw(a,c,d) != ccw(b,c,d)) and (ccw(a,b,c) != ccw(a,b,d))