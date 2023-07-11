import math
import heapdict
import numpy as np
import time
import vidmaker
from sortedcollections import OrderedSet
import pygame
import random

# Define new obstacles based on user input buffer
def obstacles_rec(obstacle_buffer, robot_size=10.5):
    obstacles = []

    buffer_val = obstacle_buffer + robot_size

    # 1st Column of Obstacles
    # Rectangle 1 obstacle space
    x1_rec1 = 70 - buffer_val
    obstacles.append(x1_rec1)

    x2_rec1 = 85 + buffer_val
    obstacles.append(x2_rec1)

    y1_rec1 = 37.5 - buffer_val
    obstacles.append(y1_rec1)

    y2_rec1 = 52.5 + buffer_val
    obstacles.append(y2_rec1)

    # Rectangle 2 obstacle space
    x1_rec2 = 70 - buffer_val
    obstacles.append(x1_rec2)

    x2_rec2 = 85 + buffer_val
    obstacles.append(x2_rec2)

    y1_rec2 = 92.5 - buffer_val
    obstacles.append(y1_rec2)

    y2_rec2 = 107.5 + buffer_val
    obstacles.append(y2_rec2)

    # Rectangle 3 obstacle space
    x1_rec3 = 70 - buffer_val
    obstacles.append(x1_rec3)

    x2_rec3 = 85 + buffer_val
    obstacles.append(x2_rec3)

    y1_rec3 = 147.5 - buffer_val
    obstacles.append(y1_rec3)

    y2_rec3 = 162.5 + buffer_val
    obstacles.append(y2_rec3)


    # 2nd Column of Obstacles
    # Rectangle 4 obstacle space
    x1_rec4 = 145 - buffer_val
    obstacles.append(x1_rec4)

    x2_rec4 = 155 + buffer_val
    obstacles.append(x2_rec4)

    y1_rec4 = 0
    obstacles.append(y1_rec4)

    y2_rec4 = 15 + buffer_val
    obstacles.append(y2_rec4)

    # Rectangle 5 obstacle space
    x1_rec5 = 145 - buffer_val
    obstacles.append(x1_rec5)

    x2_rec5 = 155 + buffer_val
    obstacles.append(x2_rec5)

    y1_rec5 = 60 - buffer_val
    obstacles.append(y1_rec5)

    y2_rec5 = 75 + buffer_val
    obstacles.append(y2_rec5)

    # Rectangle 6 obstacle space
    x1_rec6 = 145 - buffer_val
    obstacles.append(x1_rec6)

    x2_rec6 = 155 + buffer_val
    obstacles.append(x2_rec6)

    y1_rec6 = 125 - buffer_val
    obstacles.append(y1_rec6)

    y2_rec6 = 140 + buffer_val
    obstacles.append(y2_rec6)

    # Rectangle 7 obstacle space
    x1_rec7 = 145 - buffer_val
    obstacles.append(x1_rec7)

    x2_rec7 = 155 + buffer_val
    obstacles.append(x2_rec7)

    y1_rec7 = 185 - buffer_val
    obstacles.append(y1_rec7)

    y2_rec7 = 200
    obstacles.append(y2_rec7)


    # 3rd Column of Obstacles
    # Rectangle 6 obstacle space
    x1_rec8 = 220 - buffer_val
    obstacles.append(x1_rec8)

    x2_rec8 = 235 + buffer_val
    obstacles.append(x2_rec8)

    y1_rec8 = 37.5 - buffer_val
    obstacles.append(y1_rec8)

    y2_rec8 = 52.5 + buffer_val
    obstacles.append(y2_rec8)

    # Rectangle 7 obstacle space
    x1_rec9 = 220 - buffer_val
    obstacles.append(x1_rec9)

    x2_rec9 = 235 + buffer_val
    obstacles.append(x2_rec9)

    y1_rec9 = 92.5 - buffer_val
    obstacles.append(y1_rec9)

    y2_rec9 = 107.5 + buffer_val
    obstacles.append(y2_rec9)

    # Rectangle 8 obstacle space
    x1_rec10 = 220 - buffer_val
    obstacles.append(x1_rec10)

    x2_rec10 = 235 + buffer_val
    obstacles.append(x2_rec10)

    y1_rec10 = 147.5 - buffer_val
    obstacles.append(y1_rec10)

    y2_rec10 = 162.5 + buffer_val
    obstacles.append(y2_rec10)

    # Boundary obstacle space
    x1_bound = 0 + buffer_val
    x2_bound = 300 - buffer_val
    y1_bound = 0 + buffer_val
    y2_bound = 200 - buffer_val
    obstacles.append(x1_bound)
    obstacles.append(x2_bound)
    obstacles.append(y1_bound)
    obstacles.append(y2_bound)

    return obstacles

# Check if the robot is in obstacle space.
def check_obstacles(x, y):
    x1_rec1 = obstacles_var1[0]
    x2_rec1 = obstacles_var1[1]
    y1_rec1 = obstacles_var1[2]
    y2_rec1 = obstacles_var1[3]

    x1_rec2 = obstacles_var1[4]
    x2_rec2 = obstacles_var1[5]
    y1_rec2 = obstacles_var1[6]
    y2_rec2 = obstacles_var1[7]

    x1_rec3 = obstacles_var1[8]
    x2_rec3 = obstacles_var1[9]
    y1_rec3 = obstacles_var1[10]
    y2_rec3 = obstacles_var1[11]

    x1_rec4 = obstacles_var1[12]
    x2_rec4 = obstacles_var1[13]
    y1_rec4 = obstacles_var1[14]
    y2_rec4 = obstacles_var1[15]

    x1_rec5 = obstacles_var1[16]
    x2_rec5 = obstacles_var1[17]
    y1_rec5 = obstacles_var1[18]
    y2_rec5 = obstacles_var1[19]

    x1_rec6 = obstacles_var1[20]
    x2_rec6 = obstacles_var1[21]
    y1_rec6 = obstacles_var1[22]
    y2_rec6 = obstacles_var1[23]

    x1_rec7 = obstacles_var1[24]
    x2_rec7 = obstacles_var1[25]
    y1_rec7 = obstacles_var1[26]
    y2_rec7 = obstacles_var1[27]

    x1_rec8 = obstacles_var1[28]
    x2_rec8 = obstacles_var1[29]
    y1_rec8 = obstacles_var1[30]
    y2_rec8 = obstacles_var1[31]

    x1_rec9 = obstacles_var1[32]
    x2_rec9 = obstacles_var1[33]
    y1_rec9 = obstacles_var1[34]
    y2_rec9 = obstacles_var1[35]

    x1_rec10 = obstacles_var1[36]
    x2_rec10 = obstacles_var1[37]
    y1_rec10 = obstacles_var1[38]
    y2_rec10 = obstacles_var1[39]

    x1_bound = obstacles_var1[40]
    x2_bound = obstacles_var1[41]
    y1_bound = obstacles_var1[42]
    y2_bound = obstacles_var1[43]

    if (((x1_rec1) <= x <= (x2_rec1)) and ((y1_rec1) <= y <= (y2_rec1))):
        return False
    elif (((x1_rec2) <= x <= (x2_rec2)) and ((y1_rec2) <= y <= (y2_rec2))):
        return False
    elif (((x1_rec3) <= x <= (x2_rec3)) and ((y1_rec3) <= y <= (y2_rec3))):
        return False
    elif (((x1_rec4) <= x <= (x2_rec4)) and ((y1_rec4) <= y <= (y2_rec4))):
        return False
    elif (((x1_rec5) <= x <= (x2_rec5)) and ((y1_rec5) <= y <= (y2_rec5))):
        return False
    elif (((x1_rec6) <= x <= (x2_rec6)) and ((y1_rec6) <= y <= (y2_rec6))):
        return False
    elif (((x1_rec7) <= x <= (x2_rec7)) and ((y1_rec7) <= y <= (y2_rec7))):
        return False
    elif (((x1_rec8) <= x <= (x2_rec8)) and ((y1_rec8) <= y <= (y2_rec8))):
        return False
    elif (((x1_rec9) <= x <= (x2_rec9)) and ((y1_rec9) <= y <= (y2_rec9))):
        return False
    elif (((x1_rec10) <= x <= (x2_rec10)) and ((y1_rec10) <= y <= (y2_rec10))):
        return False
    elif ((x <= x1_bound) or (x>=x2_bound) or (y <= y1_bound) or (y >= y2_bound)):
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

# Pygame Visualization
def viz():
    pygame.init()
    size = [300, 200]
    video = vidmaker.Video("informed_rrt_star_obstacle_space_"+str(iterations)+"_"+str(check_radius_RRTS)+".mp4", late_export=True)
    d = obstacle_buffer + 5
    monitor = pygame.display.set_mode(size)
    pygame.display.set_caption("Informed RRT Arena")

    Done = False
    clock = pygame.time.Clock()
    while not Done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Done = True
        monitor.fill("black")

        # Start and Goal Nodes
        pygame.draw.circle(monitor, (0, 0, 255), to_pygame(init_pos, 200), 5)
        pygame.draw.circle(monitor, (0, 255, 0), to_pygame(goal_pos, 200), 5)

        # Walls
        pygame.draw.rect(monitor, "red", [0, 0, d, 200], 0)
        pygame.draw.rect(monitor, "red", [0, 0, 300, d], 0)
        pygame.draw.rect(monitor, "red", [0, 200 - d, 300, d], 0)
        pygame.draw.rect(monitor, "red", [300 - d, 0, d, 200], 0)

        # Rectangles
        # Set 1
        x, y = rec_pygame([70-d, 37.5-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([70, 37.5], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([70-d, 92.5-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([70, 92.5], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([70-d, 147.5-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([70, 147.5], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        # Set 2
        x, y = rec_pygame([145-d, 0-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([145, 0], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([145-d, 60-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([145, 60], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([145-d, 185-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([145, 185], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([145-d, 125-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([145, 125], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        # Set 3
        x, y = rec_pygame([220-d, 37.5-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([220, 37.5], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([220-d, 147.5-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([220, 147.5], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([220-d, 92.5-d], 200, 15+2*d)
        pygame.draw.rect(monitor, "red", [x, y, 15+2*d, 15+2*d], 0)
        x, y = rec_pygame([220, 92.5], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

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
        
        for i in path:
            pygame.draw.circle(monitor, (255, 0, 0), to_pygame(i, 200), 3)
            video.update(pygame.surfarray.pixels3d(monitor).swapaxes(0, 1), inverted=False)
            pygame.display.flip()
            clock.tick(10)

        pygame.display.flip()
        pygame.time.wait(4000)
        Done = True

    pygame.quit()
    # video.export(verbose=True)

def random_informed_point(major_axis):
    rand_scale = round(np.sqrt(random.uniform(0,1)),2)
    random_angle = round(np.pi * random.uniform(0,2),2)
    scaled_x = scale_factor * major_axis * rand_scale * np.cos(random_angle)
    scaled_y = scale_factor * minor_axis * rand_scale * np.sin(random_angle)
    scaled_point = np.array([scaled_x,scaled_y])
    ellipse_rotation = np.array([[np.cos(ellipse_angle), - np.sin(ellipse_angle)],
                                 [np.sin(ellipse_angle),   np.cos(ellipse_angle)]])
    rotated_point = ellipse_rotation @ scaled_point.T
    transformed_point = (int(rotated_point[0]+center[0]),int(rotated_point[1]+center[1]))
    
    if check_obstacles(transformed_point[0],transformed_point[1]):
        return transformed_point
    else:
        return None

def informed_RRTstar(node,current_best_cost):
    best_cost = current_best_cost
    backtrackSet = backtracking(node)
    for i in range(informed_iterations):
        major_axis = best_cost/2
        new_node = random_informed_point(major_axis)
        if new_node != None and new_node not in visited_nodes_track:
            if i%2 == 0:
                points.append(new_node)
            parent_node,cost = find_parent(new_node)
            if parent_node != None:
                node_records[str(new_node)] = parent_node,cost
                explored_nodes.append(new_node)
                visited_nodes_track.add(new_node)
                update_neighbors(new_node,parent_node,False)

                distance = find_distance(new_node[0],new_node[1],goal_pos[0],goal_pos[1])
                if distance < goal_radius:
                    key = node_records[str(new_node)][0]
                    temp_path = []
                    current_cost = node_records[str(new_node)][1]
                    while key != init_pos:
                        key = node_records[str(key)][0]
                        if key not in temp_path:
                            temp_path.append(key)
                        elif key==init_pos:
                            break
                        else:
                            print('Looping error')
                            print('key: ',key)
                            temp_path.append(key)
                            break
                    if current_cost < best_cost:
                        best_cost = current_cost
                        backtrackSet = temp_path
    # viz()
    return backtrackSet[::-1]

robot_size = 5

map_x = 300
map_y = 200

init_pos = (int(150), int(100))
goal_pos = (int(285), int(110))

iterations = 20000
informed_iterations = 1000

check_radius_RRTS = 10
goal_radius = 5
scale_factor = 1

node_records = {}
explored_nodes = []
visited_nodes_track = OrderedSet()
rand_points = []
backtrack = []
path = []
points = []

obstacle_buffer = 5
obstacles_var1 = obstacles_rec(obstacle_buffer,robot_size)

minor_axis = None
center = (init_pos[0]+(goal_pos[0]-init_pos[0])/2, init_pos[1]+(goal_pos[1]-init_pos[1])/2)
ellipse_angle = np.arctan2(goal_pos[1]-init_pos[1],goal_pos[0]-init_pos[0])

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
                if parent_node != None and check_goal_reach(new_node):
                    print('First path to the Goal found.')
                    node_records[str(new_node)] = parent_node,cost
                    explored_nodes.append(new_node)
                    visited_nodes_track.add(new_node)
                    update_neighbors(new_node,parent_node)

                    min_cost = find_distance(init_pos[0],init_pos[1],new_node[0],new_node[1]) + find_distance(goal_pos[0],goal_pos[1],new_node[0],new_node[1])
                    current_best_cost = node_records[str(new_node)][1]
                    minor_axis = np.sqrt(abs(current_best_cost**2 - min_cost**2))/2
                    if current_best_cost <= min_cost:
                        print('Current best cost is the optimal solution')
                        print('Backtracking path:')
                        print(backtracking(new_node))
                    else:
                        print('Informed RRT Starring now......')
                        path = informed_RRTstar(new_node,current_best_cost)
                        print(path)
                    end = time.time()
                    print('Time: ', round((end - start), 2), 's')
                    viz()
                    break
                elif parent_node != None:
                    node_records[str(new_node)] = parent_node,cost
                    explored_nodes.append(new_node)
                    visited_nodes_track.add(new_node)
                    update_neighbors(new_node,parent_node)
        check_last_iteration(i)
    print('The end')