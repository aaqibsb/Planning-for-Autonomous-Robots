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
    rand_x = random.randint(0, map_x)
    rand_y = random.randint(0, map_y)
    return (rand_x, rand_y)

def find_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def find_closest_node(node):
    min_distance = np.inf
    closest_point = (0, 0)
    for i in range(len(explored_nodes)):
        x_current = explored_nodes[i][0]
        y_current = explored_nodes[i][1]
        distance = find_distance(node[0],node[1],x_current,y_current)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x_current, y_current)
    return closest_point


def get_angle(node1, node2):
    # print('node1: ',node1)
    # print('node2: ',node2)
    if node1[0] != node2[0]:
        theta = np.rad2deg(np.arctan(abs(node1[1] - node2[1]) / abs(node1[0] - node2[0])))
        if node1[0] < node2[0] and node1[1] <= node2[1]:
            # print('theta:',theta)
            return np.round(np.deg2rad(theta),2)
        elif node1[0] > node2[0] and node1[1] <= node2[1]:
            # print('theta:',theta+90)
            return np.round(np.deg2rad(theta+90),2)
        elif node1[0] > node2[0] and node1[1] >= node2[1]:
            # print('theta:',theta+180)
            return np.round(np.deg2rad(theta+180),2)
        elif node1[0] < node2[0] and node1[1] >= node2[1]:
            # print('theta:',theta+270)
            return np.round(np.deg2rad(theta+270),2)
    else: 
        if node1[1] >= node2[1]:
            return np.round(np.deg2rad(270),2)
        elif node1[1] < node2[1]:
            return np.round(np.deg2rad(90),2)


# Custom rounding off function for coordinates
def custom_coord_round(a):
    if a - int(a) <= 0.25:
        return int(a)
    elif 0.25 < a - int(a) <= 0.75:
        return int(a) + 0.5
    elif 0.75 < a - int(a) < 1:
        return int(a) + 1

def get_new_node(node, theta, new_point):
    for i in range(1, step + 1):
        new_node = (custom_coord_round(node[0] + i * np.cos(theta)),
                    custom_coord_round(node[1] + i * np.sin(theta)))
        if not check_obstacles(new_node[0], new_node[1]):
            # print('I\'m here.')
            return None
    if new_node not in explored_nodes:
        node_records[str(new_node)] = node
        explored_nodes.append(new_node)
        visited_nodes_track.add(new_node)
        rand_points.append(new_point)
        return new_node
    else:
        return None


def check_goal_reach(x, y):
    if find_distance(x, y, goal_pos[0], goal_pos[1]) < goal_radius:
        print('Explored Nodes length:', len(explored_nodes))
        print('Goal Reached!')
        print('Backtracking path:')
        print(backtracking((x, y)))

        return True


def check_last_iteration(iter):
    if iter == iterations - 1:
        print('Explored Nodes length:', len(explored_nodes))
        print('Ran out of fuel.')


# Finding the optimal path
def backtracking(last_node):
    backtrack.append(last_node)
    key = node_records[str(last_node)]
    backtrack.append(key)
    while key != init_pos:
        key = node_records[str(key)]
        backtrack.append(key)
    return backtrack[::-1]


""" Convert coordinates into pygame coordinates """
def to_pygame(coords, height):
    return coords[0], height - coords[1]

""" Convert an object's coordinates into pygame coordinates """
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
    pygame.init()
    size = [300, 200]
    video = vidmaker.Video("rrt_"+str(iterations)+".mp4", late_export=True)
    d = obstacle_buffer + 5
    monitor = pygame.display.set_mode(size)
    pygame.display.set_caption("RRT Arena")

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
            m = visited_nodes_track[l]
            # my_string = "'" + str(m) + "'"
            # print(my_string)
            # print("HELLO")
            n = (node_records[str(m)])
            m = to_pygame(m, 200)
            n = to_pygame(n, 200)
            video.update(pygame.surfarray.pixels3d(monitor).swapaxes(0, 1), inverted=False)
            arrow(monitor, "white", (0, 0, 0), [m[0], m[1]], [n[0], n[1]], 0.5)
            pygame.display.flip()
            clock.tick(10000)
        for i in backtrack:
            pygame.draw.circle(monitor, (0, 255, 0), to_pygame(i, 200), 2)
            video.update(pygame.surfarray.pixels3d(monitor).swapaxes(0, 1), inverted=False)
            pygame.display.flip()
            clock.tick(20)

        pygame.display.flip()
        pygame.time.wait(4000)
        Done = True

    pygame.quit()
    # video.export(verbose=True)

robot_size = 10.5

map_x = 300
map_y = 200

init_pos = (int(150), int(100))
goal_pos = (int(285), int(110))
goal_radius = int(5)

iterations = 5000

node_records = {}
explored_nodes = []
visited_nodes_track = OrderedSet()
rand_points = []
backtrack = []

obstacle_buffer = 5
obstacles_var1 = obstacles_rec(obstacle_buffer,robot_size)

print('Initial position in obstacle?:', not check_obstacles(init_pos[0],init_pos[1]))
print('Final position in obstacle?:', not check_obstacles(goal_pos[0],goal_pos[1]))

step = 5

if __name__ == '__main__':
    start = time.time()
    node_records[str(init_pos)] = init_pos
    explored_nodes.append(init_pos)
    visited_nodes_track.add(init_pos)
    for i in range(iterations):
        new_point = random_point()
        closest_node = find_closest_node(new_point)
        angle = get_angle(closest_node,new_point)
        new_node = get_new_node(closest_node, angle, new_point)
        if new_node != None:
            val = check_goal_reach(new_node[0], new_node[1])
            if val:
                end = time.time()
                print('Time: ', round((end - start), 2), 's')
                viz()
                break
        check_last_iteration(i)
    print('The end')