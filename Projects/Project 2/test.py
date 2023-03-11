from queue import PriorityQueue
import matplotlib.pyplot as plt
import time

start = time.time()


x = []
y = []

x1 = []
y1 = []

obstacles = []


for i in range(0,601):
    for j in range(0,251):
        # Rectangles
        if 100 <= i <= 150 and 0 <= j <= 100:
            x1.append(i)
            y1.append(j)
        if 95 <= i <= 155 and 0 <= j <= 105:
            obstacles.append((i, j))
            x.append(i)
            y.append(j)
        if 100 <= i <= 150 and 150 <= j <= 250:
            x1.append(i)
            y1.append(j)
        if 95 <= i <= 155 and 145 <= j <= 250:
            obstacles.append((i, j))
            x.append(i)
            y.append(j)

        # Hexagon
        if int((-250/559)*i + 102950/559) <= j <= int((250/559)*i + 36800/559) and int(4323/20) <= i <= int(7677/20):
            if int((250/559)*i - 47050/559) <= j <= int((-250/559)*i + 186800/559):
                x1.append(i)
                y1.append(j)

        if int((-250/559)*i + 178.69) <= j <= int((250/559)*i + 71.3) and int(211.15) <= i <= int(388.85):
            if int((250/559)*i - 89.645) <= j <= int((-250/559)*i + 339.645):
                obstacles.append((i, j))
                x.append(i)
                y.append(j)

        # Triangle
        if int((2*i - 895)) <= j <= int(-2*i + 1145) and i >= 460:
            x1.append(i)
            y1.append(j)
        if int((2*i - 906.18)) <= j <= int(-2*i + 1156.18) and i >= 455 and 15 <= j <= 235:
            obstacles.append((i, j))
            x.append(i)
            y.append(j)


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
        print("\nPoints in obstacle space!\n")
    elif (goal_node_x,goal_node_y) in obstacles:
        print("\nPoints in obstacle space!\n")
    else:
        correct_node = True


visited_nodes = [(start_node_x,start_node_y)]
backtrack = {}
back = []
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
            if (i+1,j) not in visited_nodes and obstacles:
                if (i+1,j) not in parent_nodes:
                    move_right(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i+1,j)][0] > popped_node[0] + 1:
                        parent_nodes[(i+1,j)][0] = popped_node[0] + 1
                        parent_nodes[(i+1,j)][2] = popped_node[2]
                        backtrack[(i+1,j)] = (i,j)

        # Move Left
        if i-1 >= 0:
            if (i-1,j) not in visited_nodes and obstacles:
                if (i-1, j) not in parent_nodes:
                    move_left(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i-1,j)][0] > popped_node[0] + 1:
                        parent_nodes[(i-1,j)][0] = popped_node[0] + 1
                        parent_nodes[(i-1,j)][2] = popped_node[2]
                        backtrack[(i-1,j)] = (i, j)

        # Move Up
        if j+1 <= 250:
            if (i,j+1) not in visited_nodes and obstacles:
                if (i, j+1) not in parent_nodes:
                    move_up(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i,j+1)][0] > popped_node[0] + 1:
                        parent_nodes[(i,j+1)][0] = popped_node[0] + 1
                        parent_nodes[(i,j+1)][2] = popped_node[2]
                        backtrack[(i,j+1)] = (i, j)

        # Move Down
        if j-1 >= 0:
            if (i,j-1) not in visited_nodes and obstacles:
                if (i, j-1) not in parent_nodes:
                    move_down(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i,j-1)][0] > popped_node[0] + 1:
                        parent_nodes[(i,j-1)][0] = popped_node[0] + 1
                        parent_nodes[(i,j-1)][2] = popped_node[2]
                        backtrack[(i,j-1)] = (i, j)

        # Move Top Right
        if i+1 <= 600 and j+1 <= 250:
            if (i+1,j+1) not in visited_nodes and obstacles:
                if (i+1, j+1) not in parent_nodes:
                    move_tr(popped_node[0],popped_node[1],popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i+1,j+1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i+1,j+1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i+1,j+1)][2] = popped_node[2]
                        backtrack[(i+1,j+1)] = (i, j)

        # Move Top Left
        if i-1 >= 0 and j+1 <= 250:
            if (i-1, j+1) not in visited_nodes and obstacles:
                if (i-1, j+1) not in parent_nodes:
                    move_tl(popped_node[0], popped_node[1], popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i-1, j+1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i-1, j+1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i-1, j+1)][2] = popped_node[2]
                        backtrack[(i-1,j+1)] = (i, j)

        # Move Bottom Right
        if i+1 >= 0 and j-1 >= 0:
            if (i+1, j-1) not in visited_nodes and obstacles:
                if (i+1, j-1) not in parent_nodes:
                    move_bl(popped_node[0], popped_node[1], popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i+1, j-1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i+1, j-1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i+1, j-1)][2] = popped_node[2]
                        backtrack[(i+1,j-1)] = (i, j)

        # Move Bottom Left
        if i-1 >= 0 and j-1 >= 0:
            if (i-1, j-1) not in visited_nodes and obstacles:
                if (i-1, j-1) not in parent_nodes:
                    move_bl(popped_node[0], popped_node[1], popped_node[3][0], popped_node[3][1])
                else:
                    if parent_nodes[(i-1, j-1)][0] > popped_node[0] + 1.4:
                        parent_nodes[(i-1, j-1)][0] = popped_node[0] + 1.4
                        parent_nodes[(i-1, j-1)][2] = popped_node[2]
                        backtrack[(i-1,j-1)] = (i, j)

    else:
        # print(visited_nodes)
        print("Goal Reached!")

        path = backtracking(location)
        print(path)
        break

end = time.time()
print(end - start)

"""
plt.scatter(x, y,marker='o', s=1.5)
plt.scatter(x1, y1,marker='o',s=1.5)
# plt.scatter(x, y, marker='$\u25A0$',s=0.5)
# plt.scatter(x1, y1, marker='$\u25A0$',s=0.5)
plt.title("Arena")
plt.xlabel("X", fontweight='bold')
plt.ylabel("Y", fontweight='bold')
plt.xlim(0, 600)
plt.ylim(0, 250)
ax = plt.gca()
ax.set_facecolor("black")

for it in visited_nodes:
    plt.scatter(it[0], it[1], marker='s', color='green')
    plt.pause(0.00000025)

plt.show()
"""
