import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


def plot_graph(graph, title):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold')
    plt.title(title)
    plt.show()

def construct_network_structure(N, c, k, Pn, p_between_echo_chambers):

    
    graph = nx.erdos_renyi_graph(N, k / (N - 1))
    
    # Convert NodeView to a list of nodes
    nodes = list(graph.nodes())

    # Identify nodes in the echo chamber
    echo_chamber_nodes = random.sample(nodes, int(c * N))

    cluster_edges = [(node, neighbor) for node in echo_chamber_nodes for neighbor in (set(graph.neighbors(node)) - set(echo_chamber_nodes))]

    #num_edges_to_replace = min(int(Pn * k * E), round(Pn*len(cluster_edges)))
    num_edges_to_replace = int(Pn * len(cluster_edges))
    
    # Determine the number of edges to keep between echo chambers
    num_edges_to_keep = int(p_between_echo_chambers * num_edges_to_replace)

    # Replace edges within the cluster, keeping some between echo chambers
    edges_to_replace = random.sample(cluster_edges, num_edges_to_replace)
    edges_to_keep = random.sample(edges_to_replace, num_edges_to_keep)

    for edge in edges_to_replace:
        if graph.has_edge(*edge) and edge not in edges_to_keep:
            new_edge = (random.choice(echo_chamber_nodes), random.choice(echo_chamber_nodes))
            graph.add_edge(*new_edge)
            graph.remove_edge(*edge)
        
    return graph, echo_chamber_nodes

def simulate_cascade_with_thresholds(graph, initial_nodes, theta, Po):
    activated_nodes = set(random.sample(initial_nodes, len(initial_nodes) // 4))
    new_nodes_to_activate = set(initial_nodes)
    majority_activated_count = 0

    while new_nodes_to_activate:
        next_nodes_to_activate = set()

        for node in new_nodes_to_activate:
            neighbors = list(graph.neighbors(node))

            # Check if there are neighbors to avoid division by zero
            if neighbors:
                activated_neighbors = [neighbor for neighbor in neighbors if neighbor in activated_nodes]

                threshold = theta - (Po if node in initial_nodes else 0)

                # Check if activation threshold is met
                if len(activated_neighbors) / len(neighbors) > threshold:
                    next_nodes_to_activate.update(neighbors)

        activated_nodes.update(new_nodes_to_activate)
        new_nodes_to_activate = next_nodes_to_activate - activated_nodes

    # Check if a majority of nodes are activated
    if len(activated_nodes) / len(graph.nodes()) > 0.5:
        majority_activated_count = 1

    return majority_activated_count 

def run_model(parameters):
    Po, Pn, k, theta, c, N, steps = parameters
    print(parameters)
    results = []
    for _ in range(steps):
        graph, echo_chamber_nodes = construct_network_structure(N, c, k, Pn, 1-Pn)
        cascade_global = simulate_cascade_with_thresholds(graph, echo_chamber_nodes, theta, Po)
        results.append(cascade_global)
        

    average_success_rate = np.mean(results)
    average_network_polarization = Pn
    return average_network_polarization, average_success_rate, theta, Po

# Define the parameter space
start_Po, end_Po, num_Po_steps = 0.0, 0.0, 1
start_Pn, end_Pn, num_Pn_steps = 0.0, 0.9, 21
start_theta, end_theta, num_theta_steps = 0.15, 0.45, 10

# Define other parameters
k = 8  
c = 0.2 
N = 100  
steps = 1000 

Po_values = np.linspace(start_Po, end_Po, num_Po_steps)
Pn_values = np.linspace(start_Pn, end_Pn, num_Pn_steps)

theta_values = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45])

results = []
print(Po_values)
print(Pn_values)
print(theta_values)

parameter_combinations = [(Po, Pn, k, theta, c, N, steps) for Po in Po_values for Pn in Pn_values for theta in theta_values]
selected_parameter_combinations = random.sample(parameter_combinations, int(len(parameter_combinations)))

for params in selected_parameter_combinations:
    results.append(run_model(params))


network_polarization_values, virality_values, theta_values,_ = zip(*results)

# Sort the results based on network polarisation values
sorted_indices = np.argsort(network_polarization_values)
sorted_network_polarization = np.array(network_polarization_values)[sorted_indices]
sorted_virality = np.array(virality_values)[sorted_indices]
sorted_theta = np.array(theta_values)[sorted_indices]

plt.figure(figsize=(12, 8))

# Plot the results with colour mapping based on theta
vmin = min(sorted_theta)
vmax = max(sorted_theta)
scatter = plt.scatter(sorted_network_polarization, sorted_virality, c=sorted_theta, cmap='plasma', vmin=vmin, vmax=vmax)


unique_theta_values = np.unique(sorted_theta)
for theta in unique_theta_values:
    indices = np.where(sorted_theta == theta)[0]
    indices = sorted(indices, key=lambda i: sorted_network_polarization[i])
    color = plt.cm.plasma((theta - vmin) / (vmax - vmin)) 
    plt.plot(sorted_network_polarization[indices], sorted_virality[indices], '-o', color=color, label=f'Theta = {theta:.2f}')

plt.xlabel('Network Polarization', fontsize=16)
plt.ylabel('Virality', fontsize=16)
plt.title('Network Polarization vs. Virality', fontsize=18)


colorbar = plt.colorbar(scatter, label='Theta')

ticks = np.linspace(vmax, vmin, num=10) 
tick_labels = [str(round(v, 2)) for v in ticks[::]]  

colorbar.set_ticks(ticks)
colorbar.set_ticklabels(tick_labels)
colorbar.ax.invert_yaxis()

colorbar.set_label('Theta', size=16)
colorbar.ax.tick_params(labelsize=14)
plt.tick_params(labelsize=14)

plt.show()




# Define the parameter space for the second graph
start_Po_second, end_Po_second, num_Po_steps_second = 0.0, 0.3, 11  
theta_second = 0.25  
Po_values_second = np.linspace(start_Po_second, end_Po_second, num_Po_steps_second)  

# Create a list of parameter combinations for the second graph
parameter_combinations_second = [(Po, Pn, k, theta_second, c, N, steps) for Po in Po_values_second for Pn in Pn_values] 

results_second = []
for params in parameter_combinations_second:
    results_second.append(run_model(params))



network_polarization_values_second, virality_values_second, theta, Po_values_second = zip(*results_second)

# Sort the results based on network polarisation values for the second graph
sorted_indices_second = np.argsort(network_polarization_values_second)
sorted_network_polarization_second = np.array(network_polarization_values_second)[sorted_indices_second]
sorted_virality_second = np.array(virality_values_second)[sorted_indices_second]
sorted_Po_second = np.array(Po_values_second)[sorted_indices_second]

plt.figure(figsize=(12, 8))

# Plot the results with colour mapping based on Po for the second graph
vmin_second = min(sorted_Po_second)
vmax_second = max(sorted_Po_second)
scatter_second = plt.scatter(sorted_network_polarization_second, sorted_virality_second, c=sorted_Po_second, cmap='viridis', vmin=vmin_second, vmax=vmax_second)

unique_Po_values_second = np.unique(sorted_Po_second)
for Po in unique_Po_values_second:
    indices_second = np.where(sorted_Po_second == Po)[0]
    indices_second = sorted(indices_second, key=lambda i: sorted_network_polarization_second[i])
    color_second = plt.cm.viridis((Po - vmin_second) / (vmax_second - vmin_second))
    plt.plot(sorted_network_polarization_second[indices_second], sorted_virality_second[indices_second], '-o', color=color_second, label=f'Po = {Po:.2f}')

plt.xlabel('Network Polarization', fontsize=16)
plt.ylabel('Virality', fontsize=16)
plt.title('Network Polarization vs. Virality (Theta = 0.25)', fontsize=18)
plt.tick_params(labelsize=14)

colorbar_second = plt.colorbar(scatter_second, label='Po')
colorbar_second.set_label('Po', size=16)
colorbar_second.ax.tick_params(labelsize=14)

plt.show()