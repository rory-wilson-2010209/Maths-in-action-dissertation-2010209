import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


N_cluster = 100  
p_intra = 0.1  
G1 = nx.erdos_renyi_graph(N_cluster, p_intra, directed=True)
G2 = nx.erdos_renyi_graph(N_cluster, p_intra, directed=True)

p_inter = 0.01


# Create a list of potential inter-cluster edges
inter_cluster_edges = [(i, N_cluster + j) for i in G1.nodes() for j in G2.nodes() if np.random.random() < p_inter]

G = nx.disjoint_union(G1, G2)
G.add_edges_from(inter_cluster_edges)

for (u, v) in G.edges():
    if (u < N_cluster and v < N_cluster) or (u >= N_cluster and v >= N_cluster):
        # Intra-cluster edge

        G.edges[u, v]['weight'] = 0.5 + np.random.random() * 0.5  
    else:
        # Inter-cluster edge
        G.edges[u, v]['weight'] = np.random.random() * 0.5
        
# print(G.edges())
# print(G.get_edge_data(30,150)['weight'])

for i in G1.nodes():
    G.nodes[i]['opinion'] = np.random.uniform(0.25, 1)
for i in G2.nodes():
    G.nodes[N_cluster + i]['opinion'] = np.random.uniform(-1, -0.25)


# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, node_size=50, node_color=[G.nodes[i]['opinion'] for i in G.nodes()], cmap=plt.cm.plasma)
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.title("Polarised Network Structure", fontsize=20)
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap=plt.cm.plasma))
# cbar.set_label('Node Opinion', fontsize=16)
# cbar.ax.tick_params(labelsize=14)
# plt.axis('off')
# plt.show()



def update_state_sirs(node, G, states, beta, gamma, delta, epsilon, alpha, bot_nodes):
    current_state = states[node]
    neighbors = list(G.neighbors(node))
    if node in bot_nodes:
        return states
    if current_state == 'S':
        transmission_rate = 0
        for nbr in neighbors:
            if states[nbr] == 'I' and abs(G.nodes[node]['opinion'] - G.nodes[nbr]['opinion']) <= 0.35:
                if nbr in bot_nodes:
                    transmission_rate += G[node][nbr]['weight']+0.1
                else:
                    transmission_rate += G[node][nbr]['weight']
        if np.random.random() < beta * transmission_rate + epsilon:
            if np.random.random() < (1-alpha):
                states[node] = 'I'
            else:
                states[node] = 'R'
    elif current_state == 'I':
        if np.random.random() < gamma:
            states[node] = 'R'
    elif current_state == 'R':
        if np.random.random() < delta:
            states[node] = 'S'
    return states

def simulate_sirs_abm(G, beta, gamma, delta, epsilon, alpha, initial_infected, steps):
    N_bots = int(0.025 * N)
    bot_nodes = np.random.choice(G1.nodes(), size=N_bots, replace=False)
    states = {node: 'S' for node in G}
    #initial_infected_nodes = [1,2,199,200]
    initial_infected_nodes = []
    for node in initial_infected_nodes:
        states[node] = 'I'
    for node in bot_nodes:
        states[node] = 'I'
    
    S, I, R = [N - len(initial_infected_nodes)], [len(initial_infected_nodes)], [0]
    for step in range(steps):
        for node in G:
            states = update_state_sirs(node, G, states, beta, gamma, delta, epsilon, alpha, bot_nodes)
        S.append(sum(1 for state in states.values() if state == 'S'))
        I.append(sum(1 for state in states.values() if state == 'I'))
        R.append(sum(1 for state in states.values() if state == 'R'))

    return S, I, R

def count_affected_nodes(states):
    affected_nodes = sum(1 for state in states.values() if state in ['I', 'R'])
    return affected_nodes

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



N = 200  # Number of nodes
p = 0.1  # Probability of edge creation
G_np = nx.erdos_renyi_graph(N, p, directed=True)

for i in G1.nodes():
    G_np.nodes[i]['opinion'] = np.random.uniform(0, 1)
for i in G2.nodes():
    G_np.nodes[N_cluster + i]['opinion'] = np.random.uniform(-1, 0)





# Assign random weights to edges
for (u, v) in G_np.edges():
    G_np.edges[u,v]['weight'] = np.random.random()


# # Drawing the graph
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G_np) 
# nx.draw(G_np, pos, with_labels=False, node_size=30, node_color=[G_np.nodes[i]['opinion'] for i in G_np.nodes()], cmap=plt.cm.plasma)
# nx.draw_networkx_edges(G_np, pos, alpha=0.2)
# plt.title("Non-Polarised Network Structure", fontsize=24)
# plt.axis('off')
# cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1), cmap=plt.cm.plasma))
# cbar.set_label('Node Opinion', fontsize=18)
# cbar.ax.tick_params(labelsize=16)
# plt.show()



# Parameters for simulation
beta = 0.2  
gamma = 0.05  
delta = 0.005
epsilon = 0.000
alpha = 0.4
# Initial fraction of infected nodes
initial_infected = 3/200  
steps = 500

speed_p = []
speed = []
outbreak_size_p = []
outbreak_size = []
I_p_results = []
I_results = []

for i in range(1000):
    # SEIR simulation
    S_p, I_p, R_p = simulate_sirs_abm(G, beta, gamma, delta, epsilon, alpha, initial_infected, steps)
    effected_p = 200 - np.array(S_p)
    size_p = max(effected_p)
    index_100_p = np.where(effected_p >= 50)[0][0] if np.any(effected_p >= 50) else 0
    print(index_100_p, "polarised")
    speed_p.append(index_100_p)
    outbreak_size_p.append(size_p)
    I_p_results.append(I_p)
    
    

for i in range(1):
    # Simulate SEIRS model
    S, I, R = simulate_sirs_abm(G_np, beta, gamma, delta, epsilon, alpha, initial_infected, steps) 
    effected = 200 - np.array(S)
    size = max(effected)
    index_100 = np.where(effected >= 50)[0][0] if np.any(effected >= 50) else 0
    print(index_100, "non-polarised")
    speed.append(index_100)
    outbreak_size.append(size)
    I_results.append(I)
    
    
result_list_p = [x for x in speed_p if x != 0]
fail_scail_p = speed_p.count(0)/len(speed_p)
print(np.mean(result_list_p))
print(np.mean(outbreak_size_p))
print(fail_scail_p)
print(np.mean(result_list_p)/fail_scail_p)

result_list = [x for x in speed if x != 0]
np.mean(result_list) 
fail_scail = speed.count(0)/len(speed)
print(np.mean(result_list))
print(np.mean(outbreak_size))
print(fail_scail)
print(np.mean(result_list)/fail_scail)

print(np.mean(result_list)/np.mean(result_list_p))


#Polarised graph
bots = int(0.025*200)
plt.figure(figsize=(12, 8))
for I_p in I_p_results:
    plt.plot(np.array(I_p) - bots, color='pink', alpha=0.6)  
mean_I_p = np.mean([np.interp(range(steps), range(len(I)), I) for I in I_p_results], axis=0) -  bots
plt.plot(mean_I_p, color='red', linewidth=2)
plt.title('Polarised Network: Infected Over Time', fontsize=24)
plt.xlabel('Time Steps', fontsize=20)
plt.ylabel('Number of Infected Nodes', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 90) 
plt.show()

print(np.mean(mean_I_p[200:499]))


# Non-Polarised Graph
plt.figure(figsize=(12, 8))
for I in I_results:
    plt.plot(I, color='pink', alpha=0.6)  
mean_I = np.mean([np.interp(range(steps), range(len(I)), I) for I in I_results], axis=0)
plt.plot(mean_I, color='red', linewidth=2) 
plt.title('Non-Polarised Network: Infected Over Time', fontsize=24)
plt.xlabel('Time Steps', fontsize=20)
plt.ylabel('Number of Infected Nodes', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 125) 
plt.show()


print(np.mean(mean_I[60:499]))

calculate_total_edge_weights(G_np)
calculate_total_edge_weights(G)

k_i_sum_p = sum(G_np.nodes[node]['k_i'] for node in G_np.nodes())/200
print(k_i_sum_p)

k_i_sum = sum(G.nodes[node]['k_i'] for node in G.nodes())/200
print(k_i_sum)

