import tsplib95
import matplotlib.pyplot as plt
import pandas as pd

TEST_CASE = 1

# Import Dataset
data = tsplib95.load(f'test_cases/pr{TEST_CASE}.tsp')
nodes = list(data.get_nodes())

# Plot Routes
def plot_routes(route):
  xs = []
  ys = []
  for i in route:
    xs.append(data.node_coords[i][0])
    ys.append(data.node_coords[i][1])
  plt.clf()
  plt.plot(xs,ys,'y--')
  plt.xlabel('X Coordinates')
  plt.ylabel('Y Coordinates')
  plt.show()

def main():
  routes = pd.read_csv(f'test_results/pr{TEST_CASE}_route.csv', header=None)
  for route in routes.values:
    plot_routes(route)

if __name__ == "__main__":
  main()