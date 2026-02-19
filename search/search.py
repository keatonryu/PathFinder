import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.animation as animation

from utils import *
from grid import *

def gen_polygons(worldfilepath):
    polygons = []
    with open(worldfilepath, "r") as f:
        lines = f.readlines()
        lines = [line[:-1] for line in lines]
        for line in lines:
            polygon = []
            pts = line.split(';')
            for pt in pts:
                xy = pt.split(',')
                polygon.append(Point(int(xy[0]), int(xy[1])))
            polygons.append(polygon)
    return polygons

def neighbors(p):
    x, y = p
    return [(x, y+1), (x+1, y), (x, y-1), (x-1, y)] # up, right, down, left

def in_bounds(p):
    x, y = p # unpacks x and y from the tuple p
    return 0 <= x < 50 and 0 <= y < 50 # makes sure we stay on the 50x50 grid

def point_on_edge(ax, ay, bx, by, px, py):
    cross = (px -ax) * (by - ay) - (py - ay) * (bx - ax)
    if cross != 0:
        return False
    return (min(ax, bx) <= px <= max(ax, bx)) and (min(ay, by) <= py <= max(ay, by))

def point_on_polygon_border(p, polygon):
    px, py = p
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if point_on_edge(a.x, a.y, b.x, b.y, px, py):
            return True
    return False

def point_in_polygon(p, polygon):
    px, py = p
    inside = False
    n = len(polygon)

    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        ax, ay = a.x, a.y
        bx, by = b.x, b.y

        # does the edge cross the horizontal line y = py?
        if (a.y > py) != (b.y > py):
            x_intersect = (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x
            if px < x_intersect:
                inside = not inside

    return inside

def blocked_by_enclosure(p, epolygons):
    for polygon in epolygons:
        if point_on_polygon_border(p, polygon) or point_in_polygon(p, polygon):
            return True
    return False

# ================== BFS Search ==================

# rebuilds the path from the start to goal using the parent dictionary
def reconstruct_path(parent, start, goal):
    current = goal
    path = []

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

def bfs(start, goal, epolygons):
    frontier = Queue() # places we will explore next
    frontier.push(start)

    parent = {} # remembers where each node came from
    parent[start] = None
    
    explored = set() 
    nodes_expanded = 0

    while not frontier.isEmpty():
        current = frontier.pop()
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(parent, start, goal)
            steps = len(path) - 1
            return path, steps, nodes_expanded
        
        explored.add(current)

        for neighbor in neighbors(current):

            if not in_bounds(neighbor):
                continue

            if blocked_by_enclosure(neighbor, epolygons):
                continue

            if neighbor not in explored and neighbor not in parent:
                parent[neighbor] = current
                frontier.push(neighbor)

    return None, None, nodes_expanded

# ================== DFS Search ==================
# ================== GBFS Search ==================
# ================== A* Search ==================

if __name__ == "__main__":
    epolygons = gen_polygons('TestingGrid/world1_enclosures.txt')
    tpolygons = gen_polygons('TestingGrid/world1_turfs.txt')

    source = Point(8,10)
    dest = Point(43,45)

    fig, ax = draw_board()
    draw_grids(ax)
    draw_source(ax, source.x, source.y)  # source point
    draw_dest(ax, dest.x, dest.y)  # destination point
    
    # Draw enclosure polygons
    for polygon in epolygons:
        for p in polygon:
            draw_point(ax, p.x, p.y)
    for polygon in epolygons:
        for i in range(0, len(polygon)):
            draw_line(ax, [polygon[i].x, polygon[(i+1)%len(polygon)].x], [polygon[i].y, polygon[(i+1)%len(polygon)].y])
    
    # Draw turf polygons
    for polygon in tpolygons:
        for p in polygon:
            draw_green_point(ax, p.x, p.y)
    for polygon in tpolygons:
        for i in range(0, len(polygon)):
            draw_green_line(ax, [polygon[i].x, polygon[(i+1)%len(polygon)].x], [polygon[i].y, polygon[(i+1)%len(polygon)].y])

    #### Here call your search to compute and collect res_path

    ## res_path = [Point(24,17), Point(25,17), Point(26,17), Point(27,17),  
                ## Point(28,17), Point(28,18), Point(28,19), Point(28,20)]
    start = (source.x, source.y)
    goal = (dest.x, dest.y)

    path_tuples, steps, expanded = bfs(start, goal, epolygons)

    print("BFS steps:", steps)
    print("Nodes expanded:", expanded)

    res_path = []
    if path_tuples:
        for(x, y) in path_tuples:
            res_path.append(Point(x, y))
    
    for i in range(len(res_path)-1):
        draw_result_line(ax, [res_path[i].x, res_path[i+1].x], [res_path[i].y, res_path[i+1].y])
        plt.pause(0.1)
    
    plt.show()

