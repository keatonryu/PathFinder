import matplotlib.pyplot as plt
import numpy as np
import math

from utils import *
from grid import *

def gen_polygons(worldfilepath):
    polygons = []
    with open(worldfilepath, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()] # remove empty lines
        for line in lines:
            polygon = []
            pts = line.split(';')
            for pt in pts:
                xy = pt.split(',')
                polygon.append(Point(int(xy[0]), int(xy[1])))
            polygons.append(polygon)
    return polygons

# up, right, down, left
def neighbors(p):
    x, y = p
    return [(x, y+1), (x+1, y), (x, y-1), (x-1, y)]

# stay on the 50x50 grid boundaries
def in_bounds(p):
    x, y = p
    return 0 <= x < 50 and 0 <= y < 50 

# checks if point p lies on the edge between points a and b
def point_on_edge(ax, ay, bx, by, px, py):
    cross = (px -ax) * (by - ay) - (py - ay) * (bx - ax)
    if cross != 0:
        return False
    return (min(ax, bx) <= px <= max(ax, bx)) and (min(ay, by) <= py <= max(ay, by))

# checks if point p is on any edge of the polygon
def point_on_polygon_border(p, polygon):
    px, py = p
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if point_on_edge(a.x, a.y, b.x, b.y, px, py):
            return True
    return False

# algorithm to check if point p is inside a polygon
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

# returns true if point p is inside or on the border of any polygon in epolygons
def blocked_by_enclosure(p, epolygons):
    for polygon in epolygons:
        if point_on_polygon_border(p, polygon) or point_in_polygon(p, polygon):
            return True
    return False

# action cost
def action_cost(p, tpolygons):
    for polygon in tpolygons:
        if point_on_polygon_border(p, polygon) or point_in_polygon(p, polygon):
            return 1.5
    return 1.0

# rebuilds the path from the start to goal using the parent dictionary
def reconstruct_path(parent, goal):
    current = goal
    path = []

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path

# ================== BFS Search ==================

def bfs(start, goal, epolygons):
    todo = Queue() # FIFO structure for BFS
    todo.push(start)

    parent = {} 
    parent[start] = None
    
    explored = set() 
    nodes_expanded = 0

    while not todo.isEmpty():
        current = todo.pop()
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(parent, goal)
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
                todo.push(neighbor)

    return None, None, nodes_expanded

# ================== DFS Search ==================
def dfs(start, goal, epolygons):
    todo = Stack() # LIFO structure for DFS
    todo.push(start)

    parent = {}
    parent[start] = None

    explored = set()
    nodes_expanded = 0

    while not todo.isEmpty():
        current = todo.pop()
        if current in explored: # skip expanded nodes
            continue

        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(parent, goal)
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
                todo.push(neighbor)

    return None, None, nodes_expanded

# Pythagorean theorem (straight line distance) heuristic function
def heuristic(p, goal):
    x1, y1 = p
    x2, y2 = goal
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# ================== GBFS Search ==================

def gbfs(start, goal, epolygons, tpolygons):
    todo = PriorityQueue() # Min-heap based priority queue for GBFS
    todo.push(start, heuristic(start, goal)) # priority = h(n)

    parent = {}
    parent[start] = None

    explored = set()
    nodes_expanded = 0

    while not todo.isEmpty():
        current = todo.pop()

        if current in explored: # skip explored nodes
            continue

        explored.add(current)
        nodes_expanded += 1

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            steps = len(path) - 1

            # calculates path cost
            path_cost = sum(action_cost(path[i], tpolygons) for i in range(1, len(path)))
            return path, steps, nodes_expanded, path_cost
        
        for neighbor in neighbors(current):
            if not in_bounds(neighbor):
                continue
            if blocked_by_enclosure(neighbor, epolygons):
                continue
            if neighbor not in explored and neighbor not in parent:
                parent[neighbor] = current
                todo.push(neighbor, heuristic(neighbor, goal))

    return None, None, nodes_expanded, None

# ================== A* Search ==================

def astar(start, goal, epolygons, tpolygons):
    todo = PriorityQueue() # Min-heap based priority queue for A*

    g = {} # stores actual cost from start to each node
    g[start] = 0

    todo.push(start, g[start] + heuristic(start, goal)) # priority = g(n) + h(n), g(n) is 0 for the start node
    
    parent = {}
    parent[start] = None

    explored = set()
    nodes_expanded = 0

    while not todo.isEmpty():
        current = todo.pop()

        if current in explored: # skip explored nodes
            continue

        explored.add(current)
        nodes_expanded += 1

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            steps = len(path) - 1
            return path, steps, nodes_expanded, g[goal]
        
        for neighbor in neighbors(current):
            if not in_bounds(neighbor):
                continue
            if blocked_by_enclosure(neighbor, epolygons):
                continue
            if neighbor not in explored:
                new_g = g[current] + action_cost(neighbor, tpolygons)
                if neighbor not in g or new_g < g[neighbor]:
                    parent[neighbor] = current
                    g[neighbor] = new_g
                    todo.push(neighbor, g[neighbor] + heuristic(neighbor, goal))

    return None, None, nodes_expanded, None

if __name__ == "__main__":
    epolygons = gen_polygons('TestingGrid/world2_enclosures.txt')
    tpolygons = gen_polygons('TestingGrid/world2_turfs.txt')

    source = Point(5,5)
    dest = Point(45,46)
    start = (source.x, source.y)
    goal = (dest.x, dest.y)

    results = {
        "BFS": bfs(start, goal, epolygons),
        "DFS": dfs(start, goal, epolygons),
        "GBFS": gbfs(start, goal, epolygons, tpolygons),
        "A*": astar(start, goal, epolygons, tpolygons)
    }

    # prints resutls for each search algorithm
    for name, result in results.items():
        path_tuples, steps, expanded = result[0], result[1], result[2]
        path_cost = result[3] if len(result) > 3 else steps  # For BFS and DFS, path cost is just the number of steps
        if path_tuples is None:
            print(f"{name} - No path found. Nodes Expanded: {expanded}")
            path_cost = "N/A"
        else:
            print(f"{name} - Steps: {steps}, Nodes Expanded: {expanded}, Path Cost: {path_cost}")

        # moved everything here inside the for name loop
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

        res_path = []
        if path_tuples:
            for(x, y) in path_tuples:
                res_path.append(Point(x, y))
        
        for i in range(len(res_path)-1):
            draw_result_line(ax, [res_path[i].x, res_path[i+1].x], [res_path[i].y, res_path[i+1].y])

        ax.set_title(f"{name}")
plt.show()

