from matplotlib import pyplot as plt
import tsplib95
import random
import math
import copy

TEST_CASE = "01" # standard 1

# Import Dataset
data = tsplib95.load(f'test_cases/tc{TEST_CASE}.tsp')
nodes = list(data.get_nodes())


# Main
def main():
  y = []
  x = []
  mutation_rates = ["Tournament", "Roulette", "Ranking"]
  all_distances = []
  for i in range(0, 3):
    
    population_size = 50
    generations = 400
    mutation_rate = 0.004
    #mutation_rates.append(population_size)
    repeat = generations

    change_distances = []
    #for j in range(1):
    global data 
    global nodes
    # Import Dataset
    data = tsplib95.load(f'test_cases/tc01.tsp')
    nodes = list(data.get_nodes())
    attempt_distances, solution = genetic_algorithm(population_size=population_size, generations=generations, mutation_rate=mutation_rate, selection_mode=i)
    change_distances.append(attempt_distances)
      #plot_routes(solution, f"Map {j}, Population Size {population_size}, Mutation Rate {mutation_rate}")

    min_distances = [min(change_distances) for change_distances in change_distances]
    y.append(min_distances)
    x.append(repeat)
    all_distances.append(change_distances)

  # Plotting
  for i, mutation_rate in zip(all_distances, mutation_rates):
    avg_distances = []
    for k in range(len(i[0])):
      dists = []
      for j in range(len(i)):
        dists.append(i[j][k])
      avg_distances.append(sum(dists)/len(dists))
    plt.plot(avg_distances, label=f'Selection Mode: {mutation_rate:.3f}')
  
  plt.legend()
  plt.xlabel('Generations')
  plt.ylabel('Smaller Distance')
  plt.title('Genetic Algorithm Performance with Different Selection Mode')
  plt.show()



def plot_routes(solution, title):
  labels = []
  xs = []
  ys = []
  for city in solution:
    labels.append(city)
    xs.append(data.node_coords[city][0])
    ys.append(data.node_coords[city][1])
  xs.append(xs[0])
  ys.append(ys[0])
  plt.clf()
  plt.plot(xs,ys,'b-')
  # Add city labels on top of each point
  for i, label in enumerate(labels):
    plt.annotate(label, (xs[i], ys[i]), textcoords="offset points", xytext=(0,2), ha='center', fontsize=6)
  plt.xlabel('X Coordinates')
  plt.ylabel('Y Coordinates')
  plt.title(title)
  plt.show()


# Simulated Annealing
def genetic_algorithm(population_size: int, generations: int, mutation_rate: float, selection_mode: int) -> (list[list[int]], list[int]):

  #generate population
  population = [random.sample(nodes, len(nodes)) for _ in range(population_size)]
  
  best_solution = min(population, key=get_cost)
  best_cost = get_cost(best_solution)
  
  attempt_distances = []
  
  for _ in range(generations):
    population = envolve_population(population, mutation_rate, selection_mode)
    current_best_solution = min(population, key=get_cost)
    current_best_cost = get_cost(current_best_solution)
    if current_best_cost < best_cost:
      best_cost = current_best_cost
      best_solution = current_best_solution
    attempt_distances.append(best_cost)

  return attempt_distances, best_solution

# make the crossing
def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
  start, end = sorted(random.sample(range(len(parent1)), 2))
  child = parent1[start:end]
  child += [node for node in parent2 if node not in child]
  return child

# make the mutation
def mutate(route: list[int], mutation_rate: float):
  for i in range(len(route)):
    if random.random() < mutation_rate:
      j = random.randint(0, len(route) -1)
      route[j], route[i] = route[i], route[j]

# make the selection to 
def selection(population: list[list[int]], tournament_size: int = 10) -> list[int]:
  tournament_size = random.sample(population, tournament_size)
  return min(tournament_size, key=get_cost)

def selection_ranking(population: list[list[int]]) -> list[int]:
  sorted_population = sorted(population, key=get_cost)
  ranks = list(range(len(sorted_population), 0, -1))  # maior rank = melhor indivíduo
  total_rank = sum(ranks)
  probs = [r / total_rank for r in ranks]
  return random.choices(sorted_population, weights=probs, k=1)[0]

def selection_roulette(population: list[list[int]]) -> list[int]:
  costs = [get_cost(ind) for ind in population]
  max_cost = max(costs)
  fitness = [max_cost - c + 1 for c in costs]  # evitar fitness zero
  total_fitness = sum(fitness)
  probs = [f / total_fitness for f in fitness]
  return random.choices(population, weights=probs, k=1)[0]

def envolve_population(population: list[list[int]], mutation_rate: float, selection_mode: int) -> list[list[int]]:
  new_population = []

  for _ in range(len(population)):
    if selection_mode == 0:
      parent1 = selection(population, int(len(population)/10))
      parent2 = selection(population, int(len(population)/10))
    elif selection_mode == 1:
      parent1 = selection_roulette(population)
      parent2 = selection_roulette(population)
    else:
      parent1 = selection_ranking(population)
      parent2 = selection_ranking(population)

    child = crossover(parent1, parent2)
    mutate(child, mutation_rate)
    new_population.append(child)

  return new_population

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
    

if __name__ == "__main__":
  main()
