# import rospy
# from geometry_msgs.msg import Twist

import math
import heapdict
import numpy as np
import time
import vidmaker
from sortedcollections import OrderedSet
import pygame


# Calculate new 'C' value for new obstacle definition.
def calculate_new_c(m, c, buffer_val):
    if m > 0 and c < 0:
        c_new = c - ((buffer_val) * np.sqrt(1 + (m ** 2)))
        return c_new
    elif m < 0 and c > 0:
        if c > 300:
            c_new = c + ((buffer_val) * np.sqrt(1 + (m ** 2)))
            return c_new
        elif c < 300:
            c_new = c - ((buffer_val) * np.sqrt(1 + (m ** 2)))
            return c_new
    elif m > 0 and c > 0:
        c_new = c + ((buffer_val) * np.sqrt(1 + (m ** 2)))
        return c_new
    else:
        return None


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
    elif ((x <= x1_bound) or (y <= y1_bound) or (y >= y2_bound)):
        return False
    else:
        return True


# Custom rounding off function for angle
def custom_ang_round(b):
    if b >= 360:
        b = b % 360
    elif -360 < b < 0:
        b += 360
    elif b <= -360:
        b = b % 360 + 360
    return b


# Visited nodes threshold
def visited_nodes_threshold_check(x, y, theta):
    if visited_nodes[int(x)][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x + 1))][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x - 1))][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x))][int((y + 1))][int(theta)]:
        return False
    elif visited_nodes[int((x))][int((y - 1))][int(theta)]:
        return False
    elif visited_nodes[int((x + 1))][int((y + 1))][int(theta)]:
        return False
    elif visited_nodes[int((x - 1))][int((y + 1))][int(theta)]:
        return False
    elif visited_nodes[int((x + 1))][int((y - 1))][int(theta)]:
        return False
    elif visited_nodes[int((x - 1))][int((y - 1))][int(theta)]:
        return False

    if visited_nodes[int(x)][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x + 2))][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x - 2))][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x))][int((y + 2))][int(theta)]:
        return False
    elif visited_nodes[int((x))][int((y - 2))][int(theta)]:
        return False
    elif visited_nodes[int((x + 2))][int((y + 2))][int(theta)]:
        return False
    elif visited_nodes[int((x - 2))][int((y + 2))][int(theta)]:
        return False
    elif visited_nodes[int((x + 2))][int((y - 2))][int(theta)]:
        return False
    elif visited_nodes[int((x - 2))][int((y - 2))][int(theta)]:
        return False


    elif visited_nodes[int((x + 3))][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x - 3))][int(y)][int(theta)]:
        return False
    elif visited_nodes[int((x))][int((y + 3))][int(theta)]:
        return False
    elif visited_nodes[int((x))][int((y - 3))][int(theta)]:
        return False
    elif visited_nodes[int((x + 3))][int((y + 3))][int(theta)]:
        return False
    elif visited_nodes[int((x - 3))][int((y + 3))][int(theta)]:
        return False
    elif visited_nodes[int((x + 3))][int((y - 3))][int(theta)]:
        return False
    elif visited_nodes[int((x - 3))][int((y - 3))][int(theta)]:
        return False
    else:
        return True


# Check new node based on action set and making decisions to adding it to visited nodes list
def check_new_node(x, y, theta, total_cost, cost_to_go, cost_to_come, interim_points, interim_velocity):
    x = np.round(x, 1)
    y = np.round(y, 1)

    theta = custom_ang_round(np.round(theta, 2))
    if visited_nodes_threshold_check(x, y, theta):
        if visited_nodes[int(x)][int(y)][int(theta)] == 0:
            if (x, y, theta) in explored_nodes:
                if explored_nodes[(x, y, theta)][0] >= total_cost:
                    explored_nodes[(x, y, theta)] = total_cost, cost_to_go, cost_to_come
                    node_records[(x, y, theta)] = (pop[0][0], pop[0][1], pop[0][2]), interim_points
                    visited_nodes_track.add((x, y, theta))
                    velocity_track[(x, y, theta)] = (pop[0][0], pop[0][1], pop[0][2]), interim_velocity
                    return None
                else:
                    return None
            explored_nodes[(x, y, theta)] = total_cost, cost_to_go, cost_to_come
            node_records[(x, y, theta)] = (pop[0][0], pop[0][1], pop[0][2]), interim_points
            explored_mapping.append((x, y))
            visited_nodes_track.add((x, y, theta))
            velocity_track[(x, y, theta)] = (pop[0][0], pop[0][1], pop[0][2]), interim_velocity


# Non-holonomic constraint function
def action(RPM_L, RPM_R, pop):
    t = 0
    dt = 0.25
    R = 3.3
    L = 17.8
    x = pop[0][0]
    y = pop[0][1]

    theta = pop[0][2]
    interim_points = OrderedSet()
    interim_velocity = []

    x_new = x
    y_new = y
    theta_new = np.deg2rad(theta)

    interim_points.add((to_pygame((x_new, y_new), 200)))

    while t < 1:
        theta_new += (R / L) * (RPM_R - RPM_L) * dt * 2 * math.pi / 60
        x_new += ((R / 2) * (RPM_L + RPM_R) * np.cos((theta_new)) * dt * 2 * math.pi / 60)
        y_new += ((R / 2) * (RPM_L + RPM_R) * np.sin((theta_new)) * dt * 2 * math.pi / 60)

        temp_obs = check_obstacles(x_new, y_new)
        if not temp_obs:
            break
        interim_points.add((to_pygame((x_new, y_new), 200)))
        t = t + dt

    Ul = RPM_L * 2 * math.pi / 60
    Ur = RPM_R * 2 * math.pi / 60
    xd_new = (R / 2) * (Ul + Ur) * np.cos(theta_new)
    yd_new = (R / 2) * (Ul + Ur) * np.sin(theta_new)
    thetad_new = ((R / L) * (Ur - Ul))
    interim_velocity.append((float(xd_new / 100), float(yd_new / 100), float(thetad_new)))

    obs = check_obstacles(x_new, y_new)

    if obs:
        new_cost_to_go = 1.75 * np.sqrt(((x_new - x_f) ** 2) + ((y_new - y_f) ** 2))
        new_cost_to_come = np.sqrt(((x_new - x_s) ** 2) + ((y_new - y_s) ** 2))
        new_total_cost = new_cost_to_go + new_cost_to_come
        check_new_node(x_new, y_new, np.rad2deg(theta_new), new_total_cost, new_cost_to_go, new_cost_to_come,
                       interim_points, interim_velocity)


# Backtrack to find the optimal path
def backtracking(x, y, theta):
    backtrack.append((x, y, theta))
    key = node_records[(x, y, theta)][0]
    backtrack.append(key)
    while key != init_pos:
        key = node_records[key][0]
        backtrack.append(key)
    return backtrack[::-1]


def vel_backtracking(x, y, theta):
    vel_backtrack.append((0, 0, 0))
    key = velocity_track[(x, y, theta)]
    # print('Vel key1: ',key)
    vel_backtrack.extend(key[1])
    while key[0] != init_pos:
        key = velocity_track[key[0]]
        # print('Vel key2: ',key)
        vel_backtrack.extend(key[1])
    return vel_backtrack[::-1]


# Find intersecting coordinates based on (m,c) values
def find_intersection(m1, m2, c1, c2, a, b):
    A = np.array([[-m1, a], [-m2, b]])
    B = np.array([c1, c2])
    X = np.linalg.solve(A, B)
    return X


# Convert coordinates into pygame coordinates
def to_pygame(coords, height):
    return coords[0], height - coords[1]


# Convert an object's coordinates into pygame coordinates
def rec_pygame(coords, height, obj_height):
    return coords[0], height - coords[1] - obj_height


# def move(vel_path):
#     # Initializing a new node
#     rospy.init_node('a_star', anonymous=False)

#     #Creating a publisher that publishes velocity commands to /cmd_vel topic
#     vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

#     #Creating message of type Twist()
#     vel_msg = Twist()

#     print("Moving Robot with A-star path!")
#     rate = rospy.Rate(4.5)

#     for i in vel_path:
#         start = rospy.Time.now().to_sec()
#         while (rospy.Time.now().to_sec() - start) < 1:
#             vel_msg.linear.x = math.sqrt(i[0]**2 + i[1]**2)
#             vel_msg.angular.z = i[2]
#             print("x_dot, y_dot, theta_dot: ", i)
#             vel_pub.publish(vel_msg)
#             rate.sleep()

####Pygame Visualization####
def viz():
    pygame.init()
    size = [300, 200]
    d = obstacle_buffer + 10.5
    monitor = pygame.display.set_mode(size)
    pygame.display.set_caption("Arena")

    Done = False
    clock = pygame.time.Clock()
    while not Done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Done = True
        monitor.fill("black")

        # Walls
        pygame.draw.rect(monitor, "red", [0, 0, d, 200], 0)
        pygame.draw.rect(monitor, "red", [0, 0, 300, d], 0)
        pygame.draw.rect(monitor, "red", [0, 200 - d, 300, d], 0)
        pygame.draw.rect(monitor, "red", [300 - d, 0, d, 200], 0)

        # Rectangles
        # x, y = rec_pygame([250 - d, 0], 200, 125 + d)
        # pygame.draw.rect(monitor, "red", [x, y, 15 + 2 * d, 125 + d], 0)
        #
        # x, y = rec_pygame([150 - d, 75 - d], 200, 125 + d)
        # pygame.draw.rect(monitor, "red", [x, y, 15 + 2 * d, 125 + d], 0)

        # Set 1
        x, y = rec_pygame([75, 50], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([75, 100], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([75, 150], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        # Set 2
        x, y = rec_pygame([165, 50], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([165, 125], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        # Set 3
        x, y = rec_pygame([255, 50], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([255, 100], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        x, y = rec_pygame([255, 150], 200, 15)
        pygame.draw.rect(monitor, "orange", [x, y, 15, 15], 0)

        # Circle
        # pygame.draw.circle(monitor, "red", to_pygame((400, 110), 200), radius=50 + d)
        # pygame.draw.circle(monitor, "orange", to_pygame((400, 110), 200), radius=50)

        for i in the_path:
            pygame.draw.circle(monitor, (0, 255, 0), to_pygame(i, 200), 2)
            pygame.display.flip()
            clock.tick(20)

        pygame.display.flip()
        pygame.time.wait(4000)
        Done = True

    pygame.quit()

def velocity_output():
    # Writing VelNodes.txt
    file = open('sbadshah_Vel_Nodes.txt', 'w')
    file.write('sbadshah\n')
    file.write('Time step = 0.1\n')
    file.write("msg.angular.z \t\t msg.linear.x\n")
    for i in range(len(vel_backtrack)):
        file.write("\t")
        file.write(str(round(vel_backtrack[i][2],4)))
        file.write("\t\t\t\t\t")
        file.write(str(round(np.sqrt(vel_backtrack[i][0]**2 + vel_backtrack[i][1]**2),4)))
        file.write("\n")
    file.close()

# Global variable initialization
explored_nodes = heapdict.heapdict()
explored_mapping = []
visited_nodes = np.zeros((320, 200, 360))
visited_nodes_track = OrderedSet()
backtrack = []
vel_backtrack = []
node_records = {}
velocity_track = {}
pop = []
the_path = []
index = 0
RPM1 = 3
RPM2 = 6

obstacle_buffer = 5

obstacles_var1 = obstacles_rec(obstacle_buffer)
# obstacles_var2 = obstacles_circ(obstacle_buffer)
py_obstacles = obstacles_rec(obstacle_buffer, 10.5)

x_s = 20
y_s = 100

theta_s = 0
init_pos = (x_s, y_s, theta_s)

x_f = 300
y_f = 150
goal_pos = (x_f, y_f)

# The A* algorithm
if __name__ == '__main__':
    start = time.time()

    if check_obstacles(x_s, y_s) and check_obstacles(x_f, y_f):
        print('A-starring........')
        init_cost_to_go = round(np.sqrt(((x_s - x_f) ** 2) + ((y_s - y_f) ** 2)), 1)
        init_cost_to_come = 0
        init_total_cost = init_cost_to_come + init_cost_to_go
        explored_nodes[(x_s, y_s, theta_s)] = init_total_cost, init_cost_to_go, init_cost_to_come
        explored_mapping.append((x_s, y_s))
        while len(explored_nodes):
            pop = explored_nodes.popitem()
            index += 1
            if not (pop[0][0] >= x_f + 5):
                if visited_nodes[int(pop[0][0])][int(pop[0][1])][int(pop[0][2])] == 0:
                    visited_nodes[int(pop[0][0])][int(pop[0][1])][int(pop[0][2])] = 1

                    action(0, RPM1, pop)

                    action(RPM1, 0, pop)

                    action(RPM1, RPM1, pop)

                    action(0, RPM2, pop)

                    action(RPM2, 0, pop)

                    action(RPM2, RPM2, pop)

                    action(RPM1, RPM2, pop)

                    action(RPM2, RPM1, pop)


            else:
                print('Goal Reached!')
                print('Explored Nodes: ', list(explored_nodes.keys()))
                print('Last Pop: ', pop)
                the_path = backtracking(pop[0][0], pop[0][1], pop[0][2])
                print('Backtracking: ', the_path)
                the_vel_path = vel_backtracking(pop[0][0], pop[0][1], pop[0][2])
                # print('Vel Backtracking: ', the_vel_path)
                end = time.time()
                print('Time: ', round((end - start), 2), 's')
                print('Iterations: ', index)
                velocity_output()
                viz()
                # move(the_vel_path)
                break

        if not len(explored_nodes):
            print('No solution found.')
            print('Explored Nodes: ', list(explored_nodes.keys()))
            print('Last Pop: ', pop)
            end = time.time()
            print('Time: ', round((end - start), 2), 's')
            print('Iterations: ', index)

    elif not check_obstacles(x_s, y_s):
        print('Cannot A-star, starting node in an obstacle space.')
    elif not check_obstacles(x_f, y_f):
        print('Cannot A-star, goal node in an obstacle space.')


    