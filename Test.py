from Q__learningv2 import Q_learningv2
from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger

import csv
from scipy.stats import sem, t
from scipy import mean


experiment_type = input('experiment_type: ')    # ['node', 'target', 'MC', 'prob']
df = pd.read_csv("new_data/" + experiment_type + ".csv")
experiment_index = int(input('experiment_index: '))     #[0..5]
output_file = open("log/q_learning_basic.csv", "w")
result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
result.writeheader()
life_time = []
for nb_run in range(3):
    random.seed(nb_run)

    node_pos = list(literal_eval(df.node_pos[experiment_index]))
    list_node = []
    com_ran = df.commRange[experiment_index]
    energy = df.energy[experiment_index]
    energy_max = df.energy[experiment_index]
    prob = df.freq[experiment_index]
    nb_mc = df.nb_mc[experiment_index]
    alpha = df.q_alpha[experiment_index]
    clusters = df.charge_pos[experiment_index]
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
    net = Network(list_node=list_node, mc_list=mc_list, target=target)
    q_learning = Q_learningv2(nb_action=clusters, alpha=alpha)
    print("experiment {} #{}:\n\tnode: {}, target: {}, prob: {}, mc: {}, alpha: {}, cluster: {}".format(experiment_type, experiment_index, len(net.node), len(net.target), prob, nb_mc, q_learning.alpha, clusters))
    file_name = "log/q_learning_basic_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
    temp = net.simulate(optimizer=q_learning, file_name=file_name)
    life_time.append(temp[0])
    result.writerow({"nb run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

confidence = 0.95
h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
result.writerow({"nb run": mean(life_time), "lifetime": h, "dead_node": 0})
    