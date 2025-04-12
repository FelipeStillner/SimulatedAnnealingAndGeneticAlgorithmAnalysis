import tsplib95
import random
import math
import time
import copy
import pandas as pd

# Parameters
SAME_SOLUTION_MAX = 100 # standard 1500
SAME_COST_DIFF_MAX = 10000 # standard 150000
FUNCTION = 1 # 0: Inverse, 1: Insert, 2: Swap, 3: Swap Routes
INITIAL_TEMP = 5000 # standard 5000
ALPHA = 0.99 # standard 0.99
TEST_CASE = 1 # standard 1
REPEAT = 10 # standard 10

# Import Dataset
data = tsplib95.load(f'test_cases/pr{TEST_CASE}.tsp')
nodes = list(data.get_nodes())

# Main
def main():
  best_route_distance = []
  best_route = []
  convergence_time = []
  for _ in range(REPEAT):     
    start = time.time()
    route, route_distance = annealing(nodes)
    time_elapsed = time.time() - start
    best_route_distance.append(route_distance)
    best_route.append(route)
    convergence_time.append(time_elapsed)
    
  pd.DataFrame(best_route).to_csv(f'test_results/pr{TEST_CASE}_route.csv', index=False, header=False)
  pd.DataFrame(best_route_distance).to_csv(f'test_results/pr{TEST_CASE}_route_distance.csv', index=False, header=False)
  pd.DataFrame(convergence_time).to_csv(f'test_results/pr{TEST_CASE}_time_convergence.csv', index=False, header=False)

# Simulated Annealing
def annealing(initial_state) -> tuple[list[int], float]:
  temp = INITIAL_TEMP
  solution = initial_state
  same_solution = 0
  same_cost_diff = 0
  solution_cost = get_cost(solution)
  
  while same_solution < SAME_SOLUTION_MAX and same_cost_diff < SAME_COST_DIFF_MAX:
    neighbor = get_neighbor(solution)
    neighbor_cost = get_cost(neighbor)
    cost_diff = neighbor_cost - solution_cost
    if cost_diff >= 0:
      solution = neighbor
      solution_cost = neighbor_cost
      same_solution = 0
      if cost_diff == 0:
        same_cost_diff = 0    
      else:
        same_cost_diff +=1
    else:
      if random.uniform(0, 1) <= math.exp(float(cost_diff) / float(temp)):
        solution = neighbor
        solution_cost = neighbor_cost
        same_solution = 0
        same_cost_diff = 0
      else:
        same_solution +=1
        same_cost_diff+=1
    temp = temp*ALPHA

  return solution, 1/get_cost(solution)

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
  fitness = 1/float(distance)
  return fitness
    
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
