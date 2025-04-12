from matplotlib import pyplot as plt
import tsplib95
import random
import math
import time
import copy

# Parameters
REPEAT = 10000
FUNCTION = 0 # 0: Inverse, 1: Insert, 2: Swap, 3: Swap Routes
INITIAL_TEMP = 0.0000001 # standard 5000
ALPHA = 0.99 # standard 0.99
TEST_CASE = 1 # standard 1
ATTEMPT = 10 # standard 10

# Import Dataset
data = tsplib95.load(f'test_cases/pr{TEST_CASE}.tsp')
nodes = list(data.get_nodes())

# Main
def main():
  y = []
  x = []
  for init_function in range(0, 4):
    FUNCTION = init_function
    distances = []
    routes = []
    times = []
    for i in range(ATTEMPT):     
      start = time.time()
      route, route_distance = annealing()
      print(f'Attempt: {i}')
      time_elapsed = time.time() - start
      distances.append(route_distance)
      routes.append(route)
      times.append(time_elapsed)
    print(f'Function: {FUNCTION}')
    y.append(distances)
    x.append(FUNCTION)
  plt.boxplot(y, labels=x)
  plt.show()

# Simulated Annealing
def annealing() -> tuple[list[int], float]:
  probability = 0
  temp = INITIAL_TEMP
  solution = random.sample(nodes, len(nodes))
  solution_cost = get_cost(solution)

  for _ in range(REPEAT):
    neighbor = get_neighbor(solution)
    neighbor_cost = get_cost(neighbor)
    cost_diff = neighbor_cost - solution_cost
    if cost_diff <= 0:
      solution = neighbor
      solution_cost = neighbor_cost
    else:
      fitness = 1/float(solution_cost) - 1/float(neighbor_cost)
      probability = math.exp(-fitness/temp)
      if random.random() < probability:
        solution = neighbor
        solution_cost = neighbor_cost
    temp = temp*ALPHA
  return solution, get_cost(solution)

# Get Cost
def get_cost(state) -> float:
  distance = 0
  for i in range(len(state)):
    from_city = state[i]
    to_city = None
    if i+1 < len(state):
      to_city = state[i+1]
    else:
      to_city = state[0]
    distance += data.get_weight(from_city, to_city)
  return distance
    
# Get Neighbor
def get_neighbor(state: list[int]) -> list[int]:
  neighbor = copy.deepcopy(state)
  if FUNCTION == 0:
    inverse(neighbor)
  elif FUNCTION == 1:
    insert(neighbor)
  elif FUNCTION == 2 :
    swap(neighbor)
  else:
    swap_routes(neighbor)
  return neighbor 


# Next Neighbor Functions
## Inverses the order of cities in a route between node one and node two
def inverse(state: list[int]) -> list[int]:
  node_one = random.choice(state)
  new_list = list(filter(lambda city: city != node_one, state)) #route without the selected node one
  node_two = random.choice(new_list)
  state[min(node_one,node_two):max(node_one,node_two)] = state[min(node_one,node_two):max(node_one,node_two)][::-1]
  return state

## Inserts city at node j before node i
def insert(state: list[int]) -> list[int]:
  node_j = random.choice(state)
  state.remove(node_j)
  node_i = random.choice(state)
  index = state.index(node_i)
  state.insert(index, node_j)
  return state

## Swaps cities at positions i and j with each other
def swap(state: list[int]) -> list[int]:
  pos_one = random.choice(range(len(state)))
  pos_two = random.choice(range(len(state)))
  state[pos_one], state[pos_two] = state[pos_two], state[pos_one]
  return state

## Selects a subroute from a to b and inserts it at another position in the route
def swap_routes(state: list[int]) -> list[int]:
  subroute_a = random.choice(range(len(state)))
  subroute_b = random.choice(range(len(state)))
  subroute = state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
  del state[min(subroute_a,subroute_b):max(subroute_a, subroute_b)]
  insert_pos = random.choice(range(len(state)))
  for i in subroute:
    state.insert(insert_pos, i)
  return state

if __name__ == "__main__":
  main()
