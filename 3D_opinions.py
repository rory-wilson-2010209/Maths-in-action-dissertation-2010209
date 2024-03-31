import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def initialize_opinions(graph):
    opinions = {node: np.random.uniform(-1, 1, 2) for node in graph.nodes}
    nx.set_node_attributes(graph, opinions, 'opinion')
   
def plot_interaction_func(interact_func):
    # Plot the interaction function phi
    x_values = np.linspace(0, 2, 100)
    phi_values = interact_func(x_values)
    plt.plot(x_values, phi_values, label=interact_func)
    plt.xlabel('x')
    plt.ylabel(r'$\phi(x)$')
    plt.title('Interaction Function $\phi(x) = interact_func$')
    plt.legend()
    plt.show()


def plot_opinions(graph, time, opinions_history):
    plt.figure(figsize=(12, 12))

    # Plot opinions against time for all nodes
    if len(time) != 0:
        plt.subplot(2, 1, 1)
        for node in graph.nodes:
            line_color = plt.cm.magma((opinions_history[node][0] + 1) / 2)
            plt.plot(time, opinions_history[node], color = line_color)
           

        plt.xlabel('Time')
        plt.ylabel('Opinion Value')
        plt.title('Opinions Against Time')



def plot_opinions_3d(graph, time, opinions_history):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    for node in graph.nodes:
        if node in opinions_history and opinions_history[node]:
            x_values = [op[0] for op in opinions_history[node][:len(time)]]
            y_values = [op[1] for op in opinions_history[node][:len(time)]] 
            z_values = time

            line_color = plt.cm.magma((x_values[0] + 1) / 2)
            ax.plot(x_values, y_values, z_values, color=line_color, label=f'Node {node}')


    ax.set_xlim(1, -1)
    ax.set_ylim(-1, 1)

    ax.set_xlabel('Opinion 1')
    ax.set_ylabel('Opinion 2')
    ax.set_zlabel('Time')
    ax.set_title('Opinions Against Time in 3D', fontsize=16)
    plt.show()



#Interaction functions 
def phi_1(x):
    return np.exp(-6 * x)

def phi_2(x):
    return np.where(x < 0.4, 1, 0)

def phi_3(x):
    return (8/5) * (x - 1/2)**4 * (x + 1) * (x - 2)**2    



def phi_4(x):
    return np.piecewise(x, [x < 0.3, (0.3 <= x) & (x <= 1.1), x > 1.1],
                            [lambda x: np.exp((np.log(2)/0.3)*x)-1, 1.0, lambda x: np.exp(-5 * (x - 1.1))])

def phi_5(x):
    a = 0.1
    return -(8/5) * (x - 3/2 - a)**4 * (x - 3 - a) * (x-a)**2

def phi_6(x):
    return np.piecewise(x, [x < 0.3, (0.3 <= x) & (x <= 1.1), x > 1.1],
                            [1.0, 1/100, 0])

def phi_7(x):
    a = 0.3
    b = 0.3
    return np.piecewise(x, [x < a, (a <= x) & (x <= 2-b), x > 2-b],
                            [lambda x:-x/a+1, 0, lambda x: x/b -(2-b)/b])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Weight rate functions
def f_plus(x):
    N = 50
    return (x+(1/N)*(x**2))*(1-x)

def f_minus(x):
    return x

def continuous_opinion_dynamics(graph, phi, steps, dt, weightde, noise_strength):
    opinions_history = {node: [] for node in graph.nodes}
    time = []
    plot_opinions(graph, time, opinions_history)
    plot_interaction_func(phi)
   
    for step in range(steps):
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                k_i = graph.nodes[node]['k_i']
                sum_weighted_differences = 0

                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(node, neighbor)
                    weight = edge_data['weight'] if 'weight' in edge_data else 0.0
                    x_i = graph.nodes[node]['opinion']
                    x_j = graph.nodes[neighbor]['opinion']
                    sum_weighted_differences += weight * phi(np.abs(x_i - x_j)) * (x_j - x_i)
               
                # Apply stochastic noise
                apply_noise = np.random.rand() < phi(graph.nodes[node]['opinion'] - graph.nodes[np.random.choice(neighbors)]['opinion'])
                dW = np.sqrt(dt) * np.random.normal()
                noise = noise_strength * np.random.normal() * dW if apply_noise else 0
               
                dx_dt = (1 / k_i) * sum_weighted_differences + noise
                graph.nodes[node]['opinion'] += dx_dt * dt
                graph.nodes[node]['opinion'] = np.clip(graph.nodes[node]['opinion'], -1, 1)
                opinions_history[node].append(graph.nodes[node]['opinion'])
            else:
                graph.nodes[node]['opinion'] = graph.nodes[node]['opinion']
                opinions_history[node].append(graph.nodes[node]['opinion'])
           
            if weightde:
                for neighbor in neighbors:
                    edge = (node, neighbor)
                    x_i = graph.nodes[node]['opinion']
                    x_j = graph.nodes[neighbor]['opinion']

                    w_ij = graph.edges[edge]['weight']
                    phi_value = phi(np.abs(x_i - x_j))

                    dw_dt = phi_value * f_plus(w_ij) - (1 - phi_value) * f_minus(w_ij)
                    graph.edges[edge]['weight'] += dw_dt * dt
                   
        time.append(step * dt)

    plot_opinions(graph, time, opinions_history)


def continuous_opinion_dynamics_3d(graph, phi, steps, dt, weightde, noise_strength):
    opinions_history = {node: [] for node in graph.nodes}
    time = []
    plot_opinions_3d(graph, time, opinions_history)
    plot_interaction_func(phi)

    for step in range(steps):
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))

            if neighbors:
                k_i = graph.nodes[node]['k_i']
                sum_weighted_differences = 0

                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(node, neighbor)
                    weight = edge_data['weight'] if 'weight' in edge_data else 0.0
                    x_i = graph.nodes[node]['opinion']
                    x_j = graph.nodes[neighbor]['opinion']
                    sum_weighted_differences += weight * phi(np.abs(x_i - x_j)) * (x_j - x_i)

                apply_noise = np.random.rand() < phi(np.linalg.norm(graph.nodes[node]['opinion'] - graph.nodes[np.random.choice(neighbors)]['opinion']))
                dW = np.sqrt(dt) * np.random.normal(size=2)
                noise = noise_strength * dW if apply_noise else np.zeros(2)

                dx_dt = (1 / k_i) * sum_weighted_differences + noise
                graph.nodes[node]['opinion'] += dx_dt * dt
                graph.nodes[node]['opinion'] = np.clip(graph.nodes[node]['opinion'], -1, 1)
                opinions_history[node].append(graph.nodes[node]['opinion'])
            else:
                graph.nodes[node]['opinion'] = graph.nodes[node]['opinion']
                opinions_history[node].append(graph.nodes[node]['opinion'])

            if weightde:
                for neighbor in neighbors:
                    edge = (node, neighbor)
                    x_i = graph.nodes[node]['opinion']
                    x_j = graph.nodes[neighbor]['opinion']

                    w_ij = graph.edges[edge]['weight']
                    phi_value = phi(np.abs(x_i - x_j))

                    dw_dt = phi_value * f_plus(w_ij) - (1 - phi_value) * f_minus(w_ij)
                    graph.edges[edge]['weight'] += dw_dt * dt

        time.append(step * dt)

    plot_opinions_3d(graph, time, opinions_history)

def calculate_total_edge_weights(graph):
    for node in graph.nodes:
        total_weight = sum(graph.get_edge_data(node, neighbor).get('weight', 1.0) for neighbor in graph.neighbors(node))
        graph.nodes[node]['k_i'] = total_weight

def calculate_k_bar(graph):
    for node in graph.nodes:
        k_i_sum = sum(graph.nodes[neighbor]['k_i'] for neighbor in graph.neighbors(node))
        num_neighbors = len(list(graph.neighbors(node)))
        k_bar = (1 / num_neighbors) * k_i_sum if num_neighbors != 0 else 0
        graph.nodes[node]['k_bar'] = k_bar


graph = nx.erdos_renyi_graph(200, 0.1,directed=True)

for edge in graph.edges:
    weight = np.random.rand()
    graph.edges[edge]['weight'] = weight

initialize_opinions(graph)
calculate_total_edge_weights(graph)
calculate_k_bar(graph)

graph_copy = graph.copy()

continuous_opinion_dynamics_3d(graph, phi_2, steps=300, dt=0.1, weightde=False, noise_strength=0.2)
