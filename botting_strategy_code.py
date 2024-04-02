import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize_scalar

np.random.seed(10)

   
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
    N = 200
    number_of_bots = 10
    if len(time) != 0:
        plt.subplot(2, 1, 1)
        for node in graph.nodes():
            if graph.nodes[node]['is_bot'] == 1:
                line_color = "red"
                plt.plot(time, opinions_history[node], color = line_color, linewidth=2.5)
            else:
                line_color = plt.cm.magma((opinions_history[node][0] + 1) / 2)
                plt.plot(time, opinions_history[node], color = line_color)


        plt.xlabel('Time')
        plt.ylabel('Opinion Value')
        plt.title('Opinions Against Time')
        plt.show()


# Bounded interaction function
def phi_2(x):
    return np.where(x < 0.4, 1, 0)


def f_plus(x):
    N = 50
    return (x+(1/N)*(x**2))*(1-x)

def f_minus(x):
    return x

def continuous_opinion_dynamics(graph, phi, steps, dt, weightde, noise_strength):
    opinions_history = {node: [] for node in graph.nodes}
    time = []
    #plot_opinions(graph, time, opinions_history)
    #plot_interaction_func(phi)
   
    for step in range(steps):
        for node in graph.nodes:
            if graph.nodes[node]['is_bot'] == 1: 
                opinions_history[node].append(graph.nodes[node]['opinion'])
                continue  # Skip the rest of the loop for bot
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




        

def add_bots_with_strategy(graph, num_bots, opinion, strategy_func):
    max_node_id = max(graph.nodes) if graph.nodes else 0
    for i in range(1, num_bots + 1):
        bot_id = max_node_id + i
        graph.add_node(bot_id)
        graph.nodes[bot_id]['opinion'] = opinion
        #mark the node as a bot
        graph.nodes[bot_id]['is_bot'] = 1
        strategy_func(graph, bot_id)


def influence_neighbors_strategy(graph, bot_id):
    np.random.seed(5)
        
    filtered_nodes = [node for node in graph.nodes() if graph.nodes[node]['opinion'] >= -1]
    neighbors = np.random.choice(filtered_nodes, size=int(len(graph.nodes)*0.2), replace=False)
    
    for neighbor_id in neighbors:
        if neighbor_id != bot_id:
            graph.add_edge(neighbor_id, bot_id)  # Strong influence
            graph.edges[(neighbor_id, bot_id)]['weight'] = np.random.uniform(0,1)
            
            
def objective_function(x, graph, num_bots):
    #copy to keep same initial conditions
    graph_copy = graph.copy()
    # Add bots with the specified opinion and strategy
    add_bots_with_strategy(graph_copy, num_bots, x, influence_neighbors_strategy)
    calculate_total_edge_weights(graph_copy)
    calculate_k_bar(graph_copy)
    # Simulate
    continuous_opinion_dynamics(graph_copy, phi_2, steps=300, dt=0.1, weightde=False, noise_strength=0)


    positive_opinions_after = len([node for node, features in graph_copy.nodes(data=True) if features['opinion'] > 0.3]) - num_bots
    print(x)
    print(positive_opinions_after)
    #return negative to use the minimise function
    return -positive_opinions_after 


def objective_function_extreme(x, graph, num_bots):
    graph_copy = graph.copy()
    
    # Add bots with the specified opinion
    add_bots_with_strategy(graph_copy, num_bots, x, influence_neighbors_strategy)
    calculate_total_edge_weights(graph_copy)
    calculate_k_bar(graph_copy)
    
    # Simulate the opinion dynamics
    continuous_opinion_dynamics(graph_copy, phi_2, steps=400, dt=0.1, weightde=False, noise_strength=0)


    positive_opinions = [features['opinion'] for node, features in graph_copy.nodes(data=True) if features['opinion'] > -1]
    average_positive_opinions = sum(positive_opinions) / len(positive_opinions) if positive_opinions else 0
    print("bot opinions", x)
    print("average opinions in the network on a topic", average_positive_opinions) 
    return -average_positive_opinions

percent_increase_bot = []
opinion_shift = []

for i in range(10):
    # Create a random graph with probability p
    N_tot = 100
    bot_p = 0.1
    N = int(N_tot*(1-bot_p))
    p = 0.1
    graph = nx.erdos_renyi_graph(N, p, directed=True)
    num_bots = int(N_tot*bot_p)
    
    for edge in graph.edges:
        weight = np.random.rand()
        graph.edges[edge]['weight'] = weight
    
    for i in range(int(N/2)):
        graph.nodes[i]['opinion'] = np.random.uniform(0, 1)
        graph.nodes[i]['is_bot'] = 0
    for i in range(int(N/2)):
        graph.nodes[int(N/2) + i]['opinion'] = np.random.uniform(-1, 0)
        graph.nodes[int(N/2) + i]['is_bot'] = 0
    
    
    # Calculate total edge weights for each node
    calculate_total_edge_weights(graph)
    
    # Calculate and print k_bar for each node
    calculate_k_bar(graph)
    
    initial = -objective_function_extreme(0, graph, 0)
    print(initial)
    
    
    result = minimize_scalar(objective_function_extreme, args=(graph,num_bots,), bounds=(0, 1.0))
    
    best_x = result.x
    max_positives = -result.fun
    print(max_positives)
    
    percent_increase_bot.append(100*abs(max_positives-initial)/initial)
    opinion_shift.append(max_positives-initial)


    
