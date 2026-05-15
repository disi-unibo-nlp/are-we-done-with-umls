'''
For QA-GNN
It need for load the graph and vocab, to make sure the node ids are consistent with the ones used in the GNN model. 
We also add a new function to construct the subgraph for each QA pair,
which only keeps the 2-hop neighbors that connect both question and answer concepts, 
to reduce noise and speed up training.
'''


import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

repo_root = '..'

def load_ddb():            
    with open(f'{repo_root}/data/ddb/ddb_names.json') as f:
        all_names = json.load(f)
    with open(f'{repo_root}/data/ddb/ddb_relas.json') as f:
        all_relas = json.load(f)
    relas_lst = []
    for key, val in all_relas.items():
        relas_lst.append(val)
        
    ddb_ptr_to_preferred_name = {}
    ddb_ptr_to_name = defaultdict(list)
    ddb_name_to_ptr = {}
    for key, val in all_names.items():
        item_name = key
        item_ptr = val[0]
        item_preferred = val[1]
        if item_preferred == "1":
            ddb_ptr_to_preferred_name[item_ptr] = item_name
        ddb_name_to_ptr[item_name] = item_ptr
        ddb_ptr_to_name[item_ptr].append(item_name)
        
    return (relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name)


relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name = load_ddb()


ddb_ptr_lst, ddb_names_lst = [], []
for key, val in ddb_ptr_to_preferred_name.items():
    ddb_ptr_lst.append(key)
    ddb_names_lst.append(val)

with open(f"{repo_root}/data/ddb/vocab.txt", "w") as fout:
    for ddb_name in ddb_names_lst:
        print (ddb_name, file=fout)

with open(f"{repo_root}/data/ddb/ptrs.txt", "w") as fout:
    for ddb_ptr in ddb_ptr_lst:
        print (ddb_ptr, file=fout)

id2concept = ddb_ptr_lst

merged_relations = [
    'belongs_to_the_category_of',
    'is_a_category',
    'may_cause',
    'is_a_subtype_of',
    'is_a_risk_factor_of',
    'is_associated_with',
    'may_contraindicate',
    'interacts_with',
    'belongs_to_the_drug_family_of',
    'belongs_to_drug_super-family',
    'is_a_vector_for',
    'may_be_allelic_with',
    'see_also',
    'is_an_ingradient_of',
    'may_treat'
]

relas_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 5, "10": 6, "12": 7, "16": 8, "17": 9, "18": 10,
             "20": 11, "26": 12, "30": 13, "233": 14}

import networkx as nx

def construct_graph():
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    graph = nx.MultiDiGraph()
    attrs = set()
    for relation in relas_lst:
        subj = concept2id[relation[0]]
        obj = concept2id[relation[1]]
        rel = relas_dict[relation[2]]
        weight = 1.
        graph.add_edge(subj, obj, rel=rel, weight=weight)
        attrs.add((subj, obj, rel))
        graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
        attrs.add((obj, subj, rel + len(relation2id)))
    return concept2id, id2relation, relation2id, graph

concept2id, id2relation, relation2id, KG = construct_graph()

def load_kg():
    global cpnet, cpnet_simple
    cpnet = KG
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)

load_kg()

from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool

def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids




# The main change, we only keep the 2-hop neighbors that connect both question and answer concepts, to reduce noise and speed up training.
def concepts_to_adj_matrices_2hop_qa_pair(data):
    qc_ids, ac_ids = data

    qa_nodes = qc_ids | ac_ids

    candidates = set()
    for node in qa_nodes:
        for neighbor in set(cpnet_simple[node]):
            if neighbor not in candidates:
                candidates.add(neighbor)

    extra_nodes = set()
    for node in candidates:
        neighbors = set(cpnet_simple[node])
        touches_q = bool(neighbors & qc_ids)
        touches_a = bool(neighbors & ac_ids)
        if touches_q and touches_a:
            extra_nodes.add(node)

    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': None}


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path):
    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            if not q_ids:
                q_ids = {concept2id['31770']} 
            a_ids = set(concept2id[c] for c in dic['ac'])
            if not a_ids:
                a_ids = {concept2id['325']}
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    res = []
    for q_ids, a_ids in qa_data:
        res.append(concepts_to_adj_matrices_2hop_qa_pair((q_ids, a_ids)))

    lens = [len(e['concepts']) for e in res]
    print ('mean #nodes', int(np.mean(lens)), 'med', int(np.median(lens)), '5th', int(np.percentile(lens, 5)), '95th', int(np.percentile(lens, 95)))

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


os.system(f'mkdir -p {repo_root}/data/medqa_usmle/graph')

for fname in ["dev", "test", "train"]:
    grounded_path = f"{repo_root}/data/medqa_usmle/grounded/{fname}.grounded.jsonl"
    kg_path       = f"{repo_root}/data/ddb/ddb.graph"
    kg_vocab_path = f"{repo_root}/data/ddb/ddb_ptrs.txt"
    output_path   = f"{repo_root}/data/medqa_usmle/graph/{fname}.graph.adj.pk"

    generate_adj_data_from_grounded_concepts(grounded_path, kg_path, kg_vocab_path, output_path)