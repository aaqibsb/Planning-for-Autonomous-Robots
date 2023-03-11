from queue import PriorityQueue
import pygame
import time

start = time.time()


x = []
y = []

x1 = []
y1 = []

obstacles = []


for i in range(0,601):
    for j in range(0,251):

        # Walls
        if 0 <= i <= 5 and 0 <= j <= 250:
            obstacles.append((i, j))

        if 0 <= i <= 600 and 245 <= j <= 250:
            obstacles.append((i, j))

        if 0 <= i <= 600 and 0 <= j <= 5:
            obstacles.append((i, j))

        if 595 <= i <= 600 and 0 <= j <= 250:
            obstacles.append((i, j))

        # Rectangles
        if 95 <= i <= 155 and 0 <= j <= 105:
            obstacles.append((i, j))

        if 95 <= i <= 155 and 145 <= j <= 250:
            obstacles.append((i, j))

        # Hexagon
        if int((-250/559)*i + 178.69) <= j <= int((250/559)*i + 71.3) and int(211.15) <= i <= int(388.85):
            if int((250/559)*i - 89.645) <= j <= int((-250/559)*i + 339.645):
                obstacles.append((i, j))

        # Triangle
        if int((2*i - 906.18)) <= j <= int(-2*i + 1156.18) and i >= 455 and 15 <= j <= 235:
            obstacles.append((i, j))


start_node_x = None
start_node_y = None
goal_node_x = None
goal_node_y = None


correct_node = False
while not correct_node:
    # Start node
    start_node = input("Enter start x & y separated by space: ")
    start_node_list = start_node.split()

    start_node_x = int(start_node_list[0])
    start_node_y = int(start_node_list[1])

    # Goal node
    goal_node = input("Enter goal x & y separated by space: ")
    goal_node_list = goal_node.split()

    goal_node_x = int(goal_node_list[0])
    goal_node_y = int(goal_node_list[1])

    if (start_node_x,start_node_y) in obstacles:
        print("\nStart Points in obstacle space!\n")
    elif (goal_node_x,goal_node_y) in obstacles:
        print("\nGoal Points in obstacle space!\n")
    else:
        correct_node = True


visited_nodes = [(start_node_x,start_node_y)]
backtrack = {}
back = []
path = []
parent_nodes = {}
d = (0, 0, 0, (start_node_x,start_node_y))
node_i = 0


pq = PriorityQueue()
pq.put(d)


def move_right(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X+1,Y))
    c2c += 1
    parent_node_i = index
    node_i += 1

    backtrack[(X+1,Y)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X+1, Y))
    item = (c2c, node_i, parent_node_i, (X+1, Y))
    pq.put(item)


def move_left(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X-1,Y))
    c2c += 1
    parent_node_i = index
    node_i += 1

    backtrack[(X-1,Y)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X-1, Y))
    item = (c2c, node_i, parent_node_i, (X-1, Y))
    pq.put(item)


def move_up(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X,Y+1))
    c2c += 1
    parent_node_i = index
    node_i += 1

    backtrack[(X,Y+1)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X, Y+1))
    item = (c2c, node_i, parent_node_i, (X, Y+1))
    pq.put(item)


def move_down(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X,Y-1))
    c2c += 1
    parent_node_i = index
    node_i += 1

    backtrack[(X,Y-1)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X, Y-1))
    item = (c2c, node_i, parent_node_i, (X, Y-1))
    pq.put(item)


def move_tr(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X+1,Y+1))
    c2c += 1.4
    parent_node_i = index
    node_i += 1

    backtrack[(X+1,Y+1)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X+1, Y+1))
    item = (c2c, node_i, parent_node_i, (X+1, Y+1))
    pq.put(item)


def move_tl(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X-1,Y+1))
    c2c += 1.4
    parent_node_i = index
    node_i += 1

    backtrack[(X-1,Y+1)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X-1, Y+1))
    item = (c2c, node_i, parent_node_i, (X-1, Y+1))
    pq.put(item)


def move_br(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X+1,Y-1))
    c2c += 1.4
    parent_node_i = index
    node_i += 1

    backtrack[(X+1,Y-1)] = (X,Y)
    parent_nodes[(X, Y)] = (c2c, node_i, parent_node_i, (X+1, Y-1))
    item = (c2c, node_i, parent_node_i, (X+1, Y-1))
    pq.put(item)


def move_bl(c2c, index, X, Y):
    global node_i
    visited_nodes.append((X-1,Y-1))
    c2c += 1.4
    parent_node_i = index
    node_i += 1

    backtrack[(X-1,Y-1)] = (X,Y)
    parent_nodes[(X,Y)] = (c2c, node_i, parent_node_i, (X-1, Y-1))
    item = (c2c, node_i, parent_node_i, (X-1, Y-1))
    pq.put(item)


def backtracking(child):
    back.append(child)
    parent = backtrack[child]
    back.append(parent)
    while parent != (start_node_x,start_node_y):
        parent = backtrack[parent]
        back.append(parent)
    path = back[::-1]
    return path


# c2c, node_index, parent_node_index, child_node_location
while pq:
    popped_node = pq.get()
    location = popped_node[3]
    i,j = location

    if (i,j) != (goal_node_x,goal_node_y):
        # Move Right
        if i+1 <= 600:
            if (i+1,j) not in visited_nodes and (i+1,j) not in obstacles:
                if (i+1,j) not in parent_nodes:
                    move_right(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i+1,j)][0] > popped_node[0] + 1:
                        parent_nodes[(i+1,j)][0] = popped_node[0] + 1
                        parent_nodes[(i+1,j)][2] = popped_node[2]
                        backtrack[(i+1,j)] = (i,j)

        # Move Left
        if i-1 >= 0:
            if (i-1,j) not in visited_nodes and (i-1,j) not in obstacles:
                if (i-1, j) not in parent_nodes:
                    move_left(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i-1,j)][0] > popped_node[0] + 1:
                        parent_nodes[(i-1,j)][0] = popped_node[0] + 1
                        parent_nodes[(i-1,j)][2] = popped_node[2]
                        backtrack[(i-1,j)] = (i, j)

        # Move Up
        if j+1 <= 250:
            if (i,j+1) not in visited_nodes and (i,j+1) not in obstacles:
                if (i, j+1) not in parent_nodes:
                    move_up(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i,j+1)][0] > popped_node[0] + 1:
                        parent_nodes[(i,j+1)][0] = popped_node[0] + 1
                        parent_nodes[(i,j+1)][2] = popped_node[2]
                        backtrack[(i,j+1)] = (i, j)

        # Move Down
        if j-1 >= 0:
            if (i,j-1) not in visited_nodes and (i,j-1) not in obstacles:
                if (i, j-1) not in parent_nodes:
                    move_down(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i,j-1)][0] > popped_node[0] + 1:
                        parent_nodes[(i,j-1)][0] = popped_node[0] + 1
                        parent_nodes[(i,j-1)][2] = popped_node[2]
                        backtrack[(i,j-1)] = (i, j)

        # Move Top Right
        if i+1 <= 600 and j+1 <= 250:
            if (i+1,j+1) not in visited_nodes and (i+1,j+1) not in obstacles:
                if (i+1, j+1) not in parent_nodes:
                    move_tr(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i+1,j+1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i+1,j+1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i+1,j+1)][2] = popped_node[2]
                        backtrack[(i+1,j+1)] = (i, j)

        # Move Top Left
        if i-1 >= 0 and j+1 <= 250:
            if (i-1,j+1) not in visited_nodes and (i-1,j+1) not in obstacles:
                if (i-1, j+1) not in parent_nodes:
                    move_tl(popped_node[0], popped_node[1], popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i-1, j+1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i-1, j+1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i-1, j+1)][2] = popped_node[2]
                        backtrack[(i-1,j+1)] = (i, j)

        # Move Bottom Right
        if i+1 >= 0 and j-1 >= 0:
            if (i+1,j-1) not in visited_nodes and (i+1,j-1) not in obstacles:
                if (i+1, j-1) not in parent_nodes:
                    move_br(popped_node[0], popped_node[1], popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i+1, j-1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i+1, j-1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i+1, j-1)][2] = popped_node[2]
                        backtrack[(i+1,j-1)] = (i, j)

        # Move Bottom Left
        if i-1 >= 0 and j-1 >= 0:
            if (i-1,j-1) not in visited_nodes and (i-1,j-1) not in obstacles:
                if (i-1, j-1) not in parent_nodes:
                    move_bl(popped_node[0], popped_node[1], popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i-1, j-1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i-1, j-1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i-1, j-1)][2] = popped_node[2]
                        backtrack[(i-1,j-1)] = (i, j)

    else:
        # print(visited_nodes)
        print("Goal Reached!\n\n")

        path = backtracking(location)
        print("Path taken:\n",path)
        print("\n")

        break

end = time.time()
print("Time taken: ", end - start,"s")


""" Convert coordinates into pygame coordinates """
def to_pygame(coords, height):
    return coords[0], height - coords[1]

""" Convert an object's coords into pygame coordinates """
def rect_pygame(coords, height, obj_height):
    return coords[0], height - coords[1] - obj_height


pygame.init()
size = [600, 250]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Arena")

done = False
clock = pygame.time.Clock()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    screen.fill("black")
    x, y = rect_pygame([95, 0], 250, 105)
    pygame.draw.rect(screen, "white", [x, y, 60, 105], 0)

    x, y = rect_pygame([100, 0], 250, 100)
    pygame.draw.rect(screen, "red", [x, y, 50, 100], 0)

    x, y = rect_pygame([95, 145], 250, 105)
    pygame.draw.rect(screen, "blue", [x, y, 60, 105], 0)

    x, y = rect_pygame([100, 150], 250, 100)
    pygame.draw.rect(screen, "red", [x, y, 50, 100], 0)

    pygame.draw.rect(screen, "blue", [0, 0, 5, 250], 0)
    pygame.draw.rect(screen, "blue", [0, 0, 600, 5], 0)
    pygame.draw.rect(screen, "blue", [0, 245, 600, 5], 0)
    pygame.draw.rect(screen, "blue", [595, 0, 5, 250], 0)

    a, b = to_pygame([455, 20], 250)
    c, d = to_pygame([463, 20], 250)
    e, f = to_pygame([1031 / 2, 125], 250)
    g, h = to_pygame([463, 230], 250)
    i, j = to_pygame([455, 230], 250)
    pygame.draw.polygon(screen, "blue", ([a, b], [c, d], [e, f], [g, h], [i, j]), 0)

    a, b = to_pygame([460, 25], 250)
    c, d = to_pygame([460, 225], 250)
    e, f = to_pygame([510, 125], 250)
    pygame.draw.polygon(screen, "red", [[a, b], [c, d], [e, f]], 0)

    a, b = to_pygame([300, 2675 / 13], 250)
    c, d = to_pygame([230, 2150 / 13], 250)
    e, f = to_pygame([230, 1100 / 13], 250)
    g, h = to_pygame([300, 575  / 13], 250)
    i, j = to_pygame([370, 1100 / 13], 250)
    k, l = to_pygame([370, 2150 / 13], 250)
    pygame.draw.polygon(screen, "blue", [[a, b], [c, d], [e, f], [g, h], [i, j], [k, l]], 0)
    pygame.draw.polygon(screen, "red", ((235, 87.5), (300, 50),(365, 87.5), (365, 162.5), (300, 200), (235, 162.5)))

    for j in visited_nodes:
        pygame.draw.circle(screen, (50, 137, 131), to_pygame(j, 250), 1)
        pygame.display.flip()
        clock.tick(60)

    for i in path:
        pygame.draw.circle(screen, (255, 255, 0), to_pygame(i, 250), 1)
        pygame.display.flip()
        clock.tick(60)

    pygame.display.flip()
    pygame.time.wait(3000)
    done = True
pygame.quit()


