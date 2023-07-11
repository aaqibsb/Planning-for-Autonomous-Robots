import math
import heapdict
import numpy as np
import time
import vidmaker
from sortedcollections import OrderedSet
import pygame
import random

# Define new obstacles based on user input buffer
def obstacles_rec(obstacle_buffer, robot_size):
    obstacles = []

    buffer_val = obstacle_buffer + robot_size

    c1_rec1 = 250
    m1_rec1 = 0
    c1_rec1_new = c1_rec1 - buffer_val
    obstacles.append((m1_rec1, c1_rec1_new))

    c2_rec1 = 265
    m2_rec1 = 0
    c2_rec1_new = c2_rec1 + buffer_val
    obstacles.append((m2_rec1, c2_rec1_new))

    c3_rec1 = 125
    m3_rec1 = 0
    c3_rec1_new = c3_rec1 + buffer_val
    obstacles.append((m3_rec1, c3_rec1_new))

    c1_rec2 = 150
    m1_rec2 = 0
    c1_rec2_new = c1_rec2 - buffer_val
    obstacles.append((m1_rec2, c1_rec2_new))

    c2_rec2 = 165
    m2_rec2 = 0
    c2_rec2_new = c2_rec2 + buffer_val
    obstacles.append((m2_rec2, c2_rec2_new))

    c3_rec2 = 75
    m3_rec2 = 0
    c3_rec2_new = c3_rec2 - buffer_val
    obstacles.append((m3_rec2, c3_rec2_new))

    c1_bound = 0 + buffer_val
    c2_bound = 600 - buffer_val
    c3_bound = 0 + buffer_val
    c4_bound = 200 - buffer_val
    obstacles.append((0, c1_bound))
    obstacles.append((0, c2_bound))
    obstacles.append((0, c3_bound))
    obstacles.append((0, c4_bound))

    return obstacles

# Check obstacle space
def obstacles_circ(obstacle_buffer, robot_size):
    a = 400
    b = 110
    c = 50 + obstacle_buffer + robot_size

    return a, b, c

# Check if the robot is in obstacle space.
def check_obstacles(x, y):
    c1_rec1 = obstacles_var1[0][1]
    c2_rec1 = obstacles_var1[1][1]
    c3_rec1 = obstacles_var1[2][1]

    c1_rec2 = obstacles_var1[3][1]
    c2_rec2 = obstacles_var1[4][1]
    c3_rec2 = obstacles_var1[5][1]

    a1_circ = obstacles_var2[0]
    b1_circ = obstacles_var2[1]
    c1_circ = obstacles_var2[2]

    c1_bound = obstacles_var1[6][1]
    c2_bound = obstacles_var1[7][1]
    c3_bound = obstacles_var1[8][1]
    c4_bound = obstacles_var1[9][1]

    if (((c1_rec1) <= x <= (c2_rec1)) and (0 <= y <= (c3_rec1))):
        return False
    elif (((c1_rec2) <= x <= (c2_rec2)) and ((c3_rec2) <= y <= 200)):
        return False
    elif ((x <= c1_bound) or (x >= c2_bound) or (y <= c3_bound) or (y >= c4_bound)):
        return False
    elif ((x - a1_circ) ** 2 + (y - b1_circ) ** 2 <= c1_circ ** 2):
        return False
    else:
        return True

def random_point():
    # random_x = np.random.choice(len(map_x),4,replace=False)
    rand_x = random.randint(0, map_x)
    rand_y = random.randint(0, map_y)
    # print(rand_x,rand_y)
    if check_obstacles(rand_x,rand_y):
        return (rand_x, rand_y)
    else:
        return None

def find_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def check_goal_reach(node):
    if find_distance(node[0], node[1], goal_pos[0], goal_pos[1]) < goal_radius:
        return True
    else:
        return False
    
def get_angle(node1, node2):
    if node1[0] != node2[0]:
        theta = np.rad2deg(np.arctan(abs(node1[1] - node2[1]) / abs(node1[0] - node2[0])))
        if node1[0] < node2[0] and node1[1] <= node2[1]:
            return np.round(np.deg2rad(theta),2)
        elif node1[0] > node2[0] and node1[1] <= node2[1]:
            return np.round(np.deg2rad(theta+90),2)
        elif node1[0] > node2[0] and node1[1] >= node2[1]:
            return np.round(np.deg2rad(theta+180),2)
        elif node1[0] < node2[0] and node1[1] >= node2[1]:
            return np.round(np.deg2rad(theta+270),2)
    else: 
        if node1[1] >= node2[1]:
            return np.round(np.deg2rad(270),2)
        elif node1[1] < node2[1]:
            return np.round(np.deg2rad(90),2)
        
def check_line_obstacles(node1,node2):
    theta = get_angle(node1,node2)
    distance = int(find_distance(node1[0],node1[1],node2[0],node2[1]))
    step = 1
    for i in np.arange(step,distance,step):
        interim_node = ((node1[0] + i * np.cos(theta)),(node1[1] + i * np.sin(theta)))
        if not check_obstacles(interim_node[0], interim_node[1]):
            return False
    return True

def find_parent(node):
    min_cost = np.inf
    parent_node = None
    neighbor_nodes = find_neighbors(node)
    goal_neighbors = find_goal_neighbors()
    for neighbor in neighbor_nodes:
        if neighbor != node and neighbor not in goal_neighbors:
            val = check_line_obstacles(node,neighbor)
            if val == True :
                cost = find_distance(node[0],node[1],neighbor[0],neighbor[1])
                total_cost = cost + node_records[str(neighbor)][1]
                if total_cost < min_cost:
                    min_cost = total_cost
                    parent_node = neighbor
    if parent_node == None:
        return None,None
    else:
        return parent_node,min_cost

def find_range(node):
    rangeX = np.arange(int(node[0]) - check_radius_RRTS, 
                       int(node[0]) + check_radius_RRTS + 1, 1)
    rangeY = np.arange(int(node[1]) - check_radius_RRTS, 
                       int(node[1]) + check_radius_RRTS + 1, 1)
    return rangeX,rangeY

def find_neighbors(node):
    neighbors = set()
    neighbors_list = []
    range_x, range_y = find_range(node)
    for i in range_x:
        i = int(i)
        for j in range_y:
            j = int(j)
            coord = (i,j)
            neighbors.add(coord)
    for coord in explored_nodes:
        if coord in neighbors and coord != node:
            neighbors_list.append(coord)
    return neighbors_list

def update_neighbors(nnode,parent_node,check=False):
    neighbor_nodes = find_neighbors(nnode)
    goal_neighbors = find_goal_neighbors()
    for neighbor in neighbor_nodes:
        if neighbor != nnode and neighbor != parent_node:
            val = check_line_obstacles(nnode,neighbor)
            if val == True:
                cost = find_distance(nnode[0],nnode[1],neighbor[0],neighbor[1])
                total_cost = cost + node_records[str(nnode)][1]
                existing_cost = node_records[str(neighbor)][1]
                if total_cost < existing_cost:
                    if check:
                        print('node: ',nnode)
                        print('neighbor node before: ',neighbor)
                        print('neighbor node record before: ',node_records[str(neighbor)])
                    node_records[str(neighbor)] = nnode,total_cost
                    if check:
                        print('node: ',nnode)
                        print('neighbor node after: ',neighbor)
                        print('neighbor node record after: ',node_records[str(neighbor)])
                        print()

def check_last_iteration(iter):
    if iter == iterations - 1:
        print('Iterations Complete.')
        print('Explored Nodes length:', len(explored_nodes))
        print('Node Records Length: ',len(node_records))
        print('Completed all Iterations, now finding an optimal path - if exists....')
        goal_reg = find_goal_neighbors()
        optimal_cost = np.inf
        optimal_node = None
        for node in goal_reg:
            if node_records[str(node)][1] < optimal_cost:
                optimal_cost = node_records[str(node)][1]
                optimal_node = node
        if optimal_node != None:
            print('Backtracking path:')
            print(backtracking(optimal_node))
        else:
            print('No optimal path exists.')
        end = time.time()
        print('Time: ', round((end - start), 2), 's')
        viz()

def find_goal_range():
    rangeX = np.arange(int(goal_pos[0]) - goal_radius, 
                       int(goal_pos[0]) + goal_radius + 1, 1)
    rangeY = np.arange(int(goal_pos[1]) - goal_radius, 
                       int(goal_pos[1]) + goal_radius + 1, 1)
    return rangeX,rangeY

def find_goal_neighbors():
    goal_points = set()
    goal_neighbors = []
    range_x, range_y = find_goal_range()
    for i in range_x:
        i = int(i)
        for j in range_y:
            j = int(j)
            coord = (i,j)
            goal_points.add(coord)
    for coord in explored_nodes:
        if coord in goal_points:
            goal_neighbors.append(coord)
    return goal_neighbors

def backtracking(last_node):
    backtrack.append(last_node)
    key = node_records[str(last_node)][0]
    backtrack.append(key)
    while key != init_pos:
        key = node_records[str(key)][0]
        backtrack.append(key)
    return backtrack[::-1]

# Convert coordinates into pygame coordinates
def to_pygame(coords, height):
    return coords[0], height - coords[1]

# Convert an object's coordinates into pygame coordinates
def rec_pygame(coords, height, obj_height):
    return coords[0], height - coords[1] - obj_height

def find_intersection(m1, m2, c1, c2, a, b):
    A = np.array([[-m1, a], [-m2, b]])
    B = np.array([c1, c2])
    X = np.linalg.solve(A, B)
    return X

# Plot arrow
def arrow(screen, lcolor, tricolor, start, end, trirad):
    pygame.draw.line(screen, lcolor, start, end, 1)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pygame.draw.polygon(screen, tricolor, ((end[0]+trirad*math.sin(math.radians(rotation)), 
                                            end[1]+trirad*math.cos(math.radians(rotation))),
                                           (end[0]+trirad*math.sin(math.radians(rotation-120)),
                                            end[1]+trirad*math.cos(math.radians(rotation-120))),
                                           (end[0]+trirad*math.sin(math.radians(rotation+120)), 
                                            end[1]+trirad*math.cos(math.radians(rotation+120)))))
def viz():
    """ Pygame Visualization """
    pygame.init()
    video = vidmaker.Video("rrt_"+str(iterations)+".mp4", late_export=True)
    size = [600, 200]
    d = obstacle_buffer + robot_size
    monitor = pygame.display.set_mode(size)
    pygame.display.set_caption("RRT Arena")

    Done = False
    clock = pygame.time.Clock()
    while not Done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Done = True
        monitor.fill("black")

        # Walls
        pygame.draw.rect(monitor, "red", [0, 0, d, 200], 0)
        pygame.draw.rect(monitor, "red", [0, 0, 600, d], 0)
        pygame.draw.rect(monitor, "red", [0, 200 - d, 600, d], 0)
        pygame.draw.rect(monitor, "red", [600 - d, 0, d, 200], 0)

        # Rectangles
        x, y = rec_pygame([250 - d, 0], 200, 125 + d)
        pygame.draw.rect(monitor, "red", [x, y, 15 + 2 * d, 125 + d], 0)

        x, y = rec_pygame([150 - d, 75 - d], 200, 125 + d)
        pygame.draw.rect(monitor, "red", [x, y, 15 + 2 * d, 125 + d], 0)

        x, y = rec_pygame([250, 0], 200, 125)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 125], 0)

        x, y = rec_pygame([150, 75], 200, 125)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 125], 0)

        # Circle
        pygame.draw.circle(monitor, "red", to_pygame((400, 110), 200), radius=50 + d)
        pygame.draw.circle(monitor, "orange", to_pygame((400, 110), 200), radius=50)

        for l in range(len(visited_nodes_track) - 2):
            pygame.draw.circle(monitor, (0, 0, 255), to_pygame(init_pos, 200), 5)
            m = visited_nodes_track[l]
            n = (node_records[str(m)][0])
            m = to_pygame(m, 200)
            n = to_pygame(n, 200)
            video.update(pygame.surfarray.pixels3d(monitor).swapaxes(0, 1), inverted=False)
            arrow(monitor, "white", (0, 0, 0), [m[0], m[1]], [n[0], n[1]], 0.5)
            pygame.display.flip()
            clock.tick(10000)
        
        for i in range(len(backtrack)):
            pygame.draw.circle(monitor, (0, 0, 255), to_pygame(backtrack[-i], 200), 3)
            video.update(pygame.surfarray.pixels3d(monitor).swapaxes(0, 1), inverted=False)
            pygame.display.flip()
            clock.tick(10)

        pygame.display.flip()
        pygame.time.wait(3000)
        Done = True

    pygame.quit()
    # video.export(verbose=True)


robot_size = 5

map_x = 600
map_y = 200

init_pos = (int(15),int(15))
goal_pos = (int(300),int(170))

iterations = 40000

check_radius_RRTS = 10
goal_radius = 5

node_records = {}
explored_nodes = []
visited_nodes_track = OrderedSet()
rand_points = []
backtrack = []
path = []
points = []

obstacle_buffer = 5
obstacles_var1 = obstacles_rec(obstacle_buffer,robot_size)
obstacles_var2 = obstacles_circ(obstacle_buffer,robot_size)


print('RRT Starring....')

if not check_obstacles(init_pos[0],init_pos[1]):
    print('Start node in the obstacle space. ABORT')
if not check_obstacles(goal_pos[0],goal_pos[1]):
    print('Goal node in the obstacle space. ABORT')

if __name__ == '__main__':
    start = time.time()
    node_records[str(init_pos)] = init_pos,0
    explored_nodes.append(init_pos)
    visited_nodes_track.add(init_pos)
    for i in range(iterations):
        new_node = random_point()
        if new_node != None:
            if new_node not in visited_nodes_track:
                parent_node,cost = find_parent(new_node)
                if parent_node != None:
                    node_records[str(new_node)] = parent_node,cost
                    explored_nodes.append(new_node)
                    visited_nodes_track.add(new_node)
                    update_neighbors(new_node,parent_node)
        check_last_iteration(i)
    print('The end')