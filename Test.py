from Q__learning import Q_learning_basic
from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger

import csv
from scipy.stats import sem, t
from scipy import mean

experiment_type = input('experiment_type: ')    # ['node', 'target', 'MC', 'prob', 'package', 'cluster']
df = pd.read_csv("new_data/" + experiment_type + ".csv")
experiment_index = int(input('experiment_index: '))     #[0..6]

# experiment    ||
# ============  ||=======================================
# node          ||  [500, 550, 600*, 650, 700, 600, 600]
# target        ||  [300, 350, 400*, 450, 500, 500, 500]
# prob          ||  [0.3, 0.4, 0.5, 0.6*, 0.7, 0.8, 0.9]
# mc            ||  [2, 3*, 4, 5, 6, 7, 8]
# cluster       ||  [16, 25, 36, 49, 64, 81*, 100]
# package_size  ||  [400*, 450, 500, 550, 600, 650, 700]

output_file = open("log/q_learning_basic.csv", "w")
result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
result.writeheader()

com_ran = df.commRange[experiment_index]
prob = df.freq[experiment_index]
nb_mc = df.nb_mc[experiment_index]
alpha = df.q_alpha[experiment_index]
clusters = df.charge_pos[experiment_index]
package_size = df.package[experiment_index]

life_time = []
for nb_run in range(3):
    random.seed(nb_run)


    energy = df.energy[experiment_index]
    energy_max = df.energy[experiment_index]
    node_pos = list(literal_eval(df.node_pos[experiment_index]))
    
    list_node = []
    for i in range(len(node_pos)):
        location = node_pos[i]
        node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                    energy_thresh=0.4 * energy, prob=prob)
        list_node.append(node)
    mc_list = []
    for id in range(nb_mc):
        mc = MobileCharger(id, energy=df.E_mc[experiment_index], capacity=df.E_max[experiment_index], e_move=df.e_move[experiment_index],
                        e_self_charge=df.e_mc[experiment_index], velocity=df.velocity[experiment_index])
        mc_list.append(mc)
    target = [int(item) for item in df.target[experiment_index].split(',')]
    net = Network(list_node=list_node, mc_list=mc_list, target=target, package_size=package_size)
    q_learning = Q_learning_basic(nb_action=clusters, alpha=alpha)
    print("experiment {} #{}:\n\tnode: {}, target: {}, prob: {}, mc: {}, alpha: {}, cluster: {}, package_size: {}".format(experiment_type, experiment_index, len(net.node), len(net.target), prob, nb_mc, q_learning.alpha, clusters, package_size))
    file_name = "log/q_learning_basic_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
    temp = net.simulate(optimizer=q_learning, file_name=file_name)
    life_time.append(temp[0])
    result.writerow({"nb_run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

confidence = 0.95
h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
result.writerow({"nb_run": mean(life_time), "lifetime": h, "dead_node": 0})
    