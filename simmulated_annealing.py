from matplotlib import pyplot as plt
import tsplib95
import random
import math
import copy

TEST_CASE = 1 # standard 1

# Import Dataset
data = tsplib95.load(f'test_cases/pr{TEST_CASE}.tsp')
nodes = list(data.get_nodes())

# Main
def main():
  y = []
  x = []
  all_distances = []
  for i in range(0, 4):
    attempt = 100

    # Parameters
    init_temp = 10**-8 # < 10**-7
    last_temp = 10**-(9+i)
    repeat = 10000
    alpha = math.exp(math.log(last_temp/init_temp)/repeat)
    function = 0

    change_distances = []
    for _ in range(attempt):   
      attempt_distances = annealing(init_temp, alpha, repeat, function)
      change_distances.append(attempt_distances)
      print(f'Change: {last_temp}, Attempt: {_}')
    min_distances = [min(change_distances) for change_distances in change_distances]
    print(f'Change: {last_temp}')
    y.append(min_distances)
    x.append(last_temp)
    all_distances.append(change_distances)
  # plt.boxplot(y, tick_labels=x)
  for i in all_distances:
    avg_distances = []
    for k in range(len(i[0])):
      dists = []
      for j in range(len(i)):
        dists.append(i[j][k])
      avg_distances.append(sum(dists)/len(dists))
    plt.plot(avg_distances)
    plt.legend(x)
  plt.show()


# Simulated Annealing
def annealing(initial_temp: float, alpha: float, repeat: int, function: int) -> list[list[int]]:
  probability = 0
  temp = initial_temp
  solution = random.sample(nodes, len(nodes))
  solution_cost = get_cost(solution)

  attempt_distances = []

  for _ in range(repeat):
    neighbor = get_neighbor(solution, function)
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
    temp = temp*alpha
    attempt_distances.append(solution_cost)

  return attempt_distances

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
def get_neighbor(state: list[int], function: int) -> list[int]:
  neighbor = copy.deepcopy(state)
  if function == 0:
    inverse(neighbor)
  elif function == 1:
    insert(neighbor)
  elif function == 2 :
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
