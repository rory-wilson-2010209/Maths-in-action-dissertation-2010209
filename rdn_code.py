import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random


def initialize_opinions(graph):
    #np.random.seed(0)
    opinions = {node: np.random.uniform(-1,1) for node in graph.nodes}
    nx.set_node_attributes(graph, opinions, 'opinion')
   
def plot_interaction_func(interact_func):
    # Plot the interaction function phi
    x_values = np.linspace(0, 2, 150)
    phi_values = interact_func(x_values)
    plt.plot(x_values, phi_values, label=interact_func, color = 'tomato')
    plt.xlabel('x', fontsize=14)
    plt.ylabel(r'$\phi(x)$', fontsize=14)
    plt.title('Interaction Function', fontsize=16)
    #plt.legend()
    plt.show()


def plot_opinions(graph, time, opinions_history):
    plt.figure(figsize=(12, 12))

   
    # Plot opinions against time for all nodes
    if len(time) != 0:
        plt.subplot(2, 1, 1)
        for node in graph.nodes:
            line_color = plt.cm.magma((opinions_history[node][0] + 1) / 2)
            plt.plot(time, opinions_history[node], color=line_color)
           
        plt.ylim(-1, 1)
        plt.xlabel('Time', fontsize=14) 
        plt.ylabel('Opinion Value', fontsize=14)
        plt.title('Opinions Against Time', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        plt.show()

    #Graph layout visualisation
    ax2 = plt.subplot(2, 1, 2)
    opinions = {node: graph.nodes[node].get('opinion', 0) for node in graph.nodes}

    # Sort nodes based on opinions
    sorted_nodes = sorted(graph.nodes, key=lambda node: opinions[node])

    num_nodes = len(graph.nodes)
    pos = {}
    # Point on the left for positive opinion nodes
    left_point = (1, 0)
    # Point on the right for negative opinion nodes
    right_point = (-1, 0)

    for i, node in enumerate(sorted_nodes):
        if opinions[node] >= 0:
            # Place positive opinion nodes closer to the left_point
            x = left_point[0] - (1 - abs(opinions[node]))  
        else:
            # Place negative opinion nodes closer to the right_point
            x = right_point[0] + (1 - abs(opinions[node]))

        
        total_outward_weight = sum(graph.get_edge_data(node, neighbor)["weight"] for neighbor in graph.neighbors(node))
        y = total_outward_weight / 100 
        pos[node] = (x, y)

    # Normalise opinion values [-1, 1]
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(vmin=-1, vmax=1)
    node_colors = [cmap(norm(opinions[node])) for node in sorted_nodes]
    node_sizes = [abs(opinions[node]) * 100 for node in sorted_nodes]

    # Calculate edge alpha based on the distance between nodes
    edge_alpha = [1/np.exp(np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)) for u, v in graph.edges()]
    max_alpha = max(edge_alpha)
    edge_alpha = [alpha / max_alpha for alpha in edge_alpha]  
    edge_alpha = [0 if element < 0.41 else element for element in edge_alpha]
    edge_alpha = np.clip(0,1000,edge_alpha)

    nx.draw_networkx_nodes(graph, pos, nodelist=sorted_nodes, node_size=node_sizes, node_color=node_colors,
                           cmap=cmap, alpha=1, linewidths=2.0, node_shape='o')

    # Draw edges
    edge_width = 1.4
    nx.draw_networkx_edges(graph, pos, width=edge_width, edge_color='black', alpha=0.2)

    nx.draw_networkx_labels(graph, pos, labels={node: '' for node in graph.nodes}, font_size=10, font_color='black')
    plt.title('Network of nodes spatially dependent on current opinion', fontsize=18)

    
    # Create colourbar axis
    sm = ScalarMappable(cmap=plt.cm.plasma)
    sm.set_clim(vmin=-1, vmax=1)
    sm.set_array(node_colors)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="2%", pad=0.1)
    cbar = plt.colorbar(sm, cax=cax, label='Opinion Value', orientation="horizontal")
    cbar.set_ticks([-1, 0, 1])
    
    plt.tight_layout()
    plt.show()


       
   
#Interaction functions 
   
def phi_1(x):
    return np.exp(-6 * x)

def phi_2(x):
    return np.where(x < 0.4, 1, 0)

def phi_3(x):
    return (8/5) * (x - 1/2)**4 * (x + 1) * (x - 2)**2    

def phi_4(x):
    return np.piecewise(x, [x < 0.45, (0.45 <= x) & (x <= 0.8), (0.8 < x) & (x <= 1.2), x > 1.2],
                            [lambda x: np.exp(-6*x), 0, 0, 0])

def phi_5(x):
    a = 0.1
    return -(8/5) * (x - 3/2 - a)**4 * (x - 3 - a) * (x-a)**2

def phi_6(x):
    return np.piecewise(x, [x <= 0.3, x > 0.3],
                            [lambda x: 1/(x+1)**4 , 0])

def phi_7(x):
    a = 0.3
    b = 0.3
    return np.piecewise(x, [x < a, (a <= x) & (x <= 2-b), x > 2-b],
                            [lambda x:-x/a+1, 0, lambda x: x/b -(2-b)/b])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_plus(x):
    N = 50
    return (x+(1/N)*(x**2))*(1-x)

def f_minus(x):
    return x

def continuous_opinion_dynamics(graph, phi, steps, dt, weightde, noise_strength, RDN, nudge_strength):
    opinions_history = {node: [] for node in graph.nodes}
    time = []
    plot_opinions(graph, time, opinions_history)
    plot_interaction_func(phi)
   
    for step in range(steps):
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node)) 
            mean_opinion = np.mean([graph.nodes[node]['opinion'] for node in graph.nodes])

            if neighbors:

                polariser = 0
                k_i = graph.nodes[node]['k_i']
                sum_weighted_differences = 0

                for neighbor in neighbors:
                    edge_data = graph.get_edge_data(node, neighbor)
                    weight = edge_data['weight'] if 'weight' in edge_data else 0.0
                    x_i = graph.nodes[node]['opinion']
                    x_j = graph.nodes[neighbor]['opinion']
                    sum_weighted_differences += weight * phi(np.abs(x_i - x_j)) * (x_j - x_i)
                
                #Random dnymaical nudge code
                if RDN == True:
                    num_random_nodes = 5
                    random_nodes = random.sample(list(graph.nodes), num_random_nodes)
                    mean_opinion_random_nodes = np.mean([graph.nodes[node]['opinion'] for node in random_nodes])
                    polariser = 2*np.sqrt(num_random_nodes)*(mean_opinion_random_nodes - mean_opinion)
               
                # Apply stochastic noise
                apply_noise = np.random.rand() < phi(np.abs(graph.nodes[node]['opinion'] - graph.nodes[np.random.choice(neighbors)]['opinion']))
                #apply_noise = True
                dW = np.sqrt(dt) * np.random.normal()
                noise = noise_strength * dW if apply_noise else 0
               
                dx_dt = (1 / k_i) * sum_weighted_differences + noise + polariser*nudge_strength
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


def plot_final_opinion_distribution(final_opinions):
    plt.figure(figsize=(8, 6))
    plt.hist(final_opinions, bins=60, edgecolor='black', alpha=0.7, color = 'tomato')
    plt.xlabel('Final Opinion Value', fontsize=14)  
    plt.ylabel('Frequency', fontsize=14)          
    plt.title('Distribution of Final Opinions', fontsize=16) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-1, 1)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.show()

# Erdos renyi graph generation 
graph = nx.erdos_renyi_graph(200, 0.2, directed=True)


#np.random.seed(42)
for edge in graph.edges:
    weight = np.random.rand()
    graph.edges[edge]['weight'] = weight
    
# Initialize random opinions for each node
initialize_opinions(graph)

# Calculatr total edge weights (k_i) for each node
calculate_total_edge_weights(graph)

# Calculate and print k_bar for each node
calculate_k_bar(graph)

graphdupe5 = graph.copy()

# Simulate
continuous_opinion_dynamics(graphdupe5, phi_2, steps=700, dt=0.1, weightde=False, noise_strength=0, RDN=False, nudge_strength=0)
final_opinions = [graphdupe5.nodes[node]['opinion'] for node in graph.nodes]
plot_final_opinion_distribution(final_opinions)


graphdupe1 = graphdupe5.copy()
graphdupe2 = graphdupe5.copy()
graphdupe3 = graphdupe5.copy()
graphdupe4 = graphdupe5.copy()
graphdupe6 = graphdupe5.copy()


continuous_opinion_dynamics(graphdupe1, phi_2, steps=700, dt=0.1, weightde=False, noise_strength=1, RDN=False, nudge_strength=1)
final_opinions = [graphdupe1.nodes[node]['opinion'] for node in graph.nodes]
plot_final_opinion_distribution(final_opinions)


continuous_opinion_dynamics(graphdupe2, phi_2, steps=700, dt=0.1, weightde=False, noise_strength=1, RDN=True, nudge_strength=0.5)
final_opinions = [graphdupe2.nodes[node]['opinion'] for node in graph.nodes]
plot_final_opinion_distribution(final_opinions)

continuous_opinion_dynamics(graphdupe3, phi_2, steps=700, dt=0.1, weightde=False, noise_strength=1, RDN=True, nudge_strength=0.75)
final_opinions = [graphdupe3.nodes[node]['opinion'] for node in graph.nodes]
plot_final_opinion_distribution(final_opinions)


continuous_opinion_dynamics(graphdupe4, phi_2, steps=700, dt=0.1, weightde=False, noise_strength=1, RDN=True, nudge_strength=1)
final_opinions = [graphdupe4.nodes[node]['opinion'] for node in graph.nodes]
plot_final_opinion_distribution(final_opinions)


continuous_opinion_dynamics(graphdupe6, phi_2, steps=700, dt=0.1, weightde=False, noise_strength=1, RDN=True, nudge_strength=1.25)
final_opinions = [graphdupe6.nodes[node]['opinion'] for node in graph.nodes]
plot_final_opinion_distribution(final_opinions)
