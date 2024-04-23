#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:17:04 2023
@author: minjcha
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, cdist
import itertools
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from itertools import combinations
import math
import heapq
import cvxpy as cvx
import networkit as nk
import time
import collections
import copy

def nodes(xyz):
    nodes = np.array(xyz.index)
    return nodes

def edges(xyz, cutoff, tol):
    c1 = xyz.iloc[:, 1:].to_numpy()
    dists = cdist(c1, c1)
    intlocs = np.array(np.where((dists <= cutoff * tol) & (dists > 0))).T
    sorted_arr = np.sort(intlocs, axis=1)
    edges = np.unique(sorted_arr, axis=0)

    weights = dists[edges[:, 0], edges[:, 1]]

    return edges, weights

def ollivier_ricci_node(G, alpha):
   
    orc = OllivierRicci(G, alpha=alpha, verbose="INFO", weight = "weight")
    orc.compute_ricci_curvature()
    c_sum = []
    for ii in G.nodes:
        L = list(orc.G[ii].values())
        c = []
        for jj in range(len(L)):
            c.append(L[jj]['ricciCurvature'])
        
        c_sum.append(np.sum(c))
        
    return c_sum

def ollivier_ricci_edge(G, alpha):
  
    orc = OllivierRicci(G, alpha=alpha, verbose="INFO", weight = "weight")
    orc.compute_ricci_curvature()
    
    edge_curve = []
    for ii in range(len(list(G.edges))):
        n1 = list(G.edges)[ii][0]
        n2 = list(G.edges)[ii][1]
        edge_curve.append(orc.G[n1][n2]['ricciCurvature'])
    
    return edge_curve

def node_connect_path(g, edge):
    common = np.sort(list(set(g.neighbors(edge[0])).intersection(g.neighbors(edge[1])).difference(edge)))
    connect = np.array(list(combinations(common, 2)))

    path = pd.DataFrame(np.zeros((len(connect), 4)), columns=['e0', 'e1', 'c0', 'c1'], dtype='int32')
    if len(path) > 0:
        path['e0'] = edge[0]
        path['e1'] = edge[1]
        path['c0'] = connect[:, 0]
        path['c1'] = connect[:, 1]

    return common, connect, path

def path_sum_mean(edge, path):
    path_sum_mean = pd.DataFrame(np.zeros((len(edge), 6)),
                                 columns=['e0', 'e1', 'op_sum', 'op_mean', 'sp_sum', 'sp_mean'])

    for ii in range(len(path)):
        path_sum_mean['e0'].iloc[ii] = edge[ii][0]
        path_sum_mean['e1'].iloc[ii] = edge[ii][1]
        path_sum_mean['op_sum'].iloc[ii] = path[ii]['op_c0-c1'].sum()
        path_sum_mean['op_mean'].iloc[ii] = path[ii]['op_c0-c1'].mean()
        path_sum_mean['sp_sum'].iloc[ii] = path[ii]['sp_c0-c1'].sum()
        path_sum_mean['sp_mean'].iloc[ii] = path[ii]['sp_c0-c1'].mean()

    return path_sum_mean

_base = math.e
_exp_power = 2
_alpha = 0.5
_nbr_topk = 1000
EPSILON = 1e-7 

def get_single_node_neighbors_distributions(Gk, node, direction="successors"):
    """Get the neighbor density distribution of given node `node`.
    Parameters
    ----------
    node : int
        Node index in Networkit graph `_Gk`.
    direction : {"predecessors", "successors"}
        Direction of neighbors in directed graph. (Default value: "successors")
    Returns
    -------
    distributions : lists of float
        Density distributions of neighbors up to top `_nbr_topk` nodes.
    nbrs : lists of int
        Neighbor index up to top `_nbr_topk` nodes.
    """
    if Gk.isDirected():
        if direction == "predecessors":
            neighbors = [_ for _ in Gk.iterInNeighbors(node)]
        else:  # successors
            neighbors = [_ for _ in Gk.iterNeighbors(node)]
    else:
        neighbors = [_ for _ in Gk.iterNeighbors(node)]

    # Get sum of distributions from x's all neighbors
    heap_weight_node_pair = []
    for nbr in neighbors:
        if direction == "predecessors":
            w = _base ** (-Gk.weight(nbr, node) ** _exp_power)
        else:  # successors
            w = _base ** (-Gk.weight(node, nbr) ** _exp_power)

        if len(heap_weight_node_pair) < _nbr_topk:
            heapq.heappush(heap_weight_node_pair, (w, nbr))
        else:
            heapq.heappushpop(heap_weight_node_pair, (w, nbr))

    nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])

    if len(neighbors) == 0:
        # No neighbor, all mass stay at node
        return [1], [node]
    elif nbr_edge_weight_sum > EPSILON:
        # Sum need to be not too small to prevent divided by zero
        distributions = [(1.0 - _alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]
    else:
        # Sum too small, just evenly distribute to every neighbors
        logger.warning("Neighbor weight sum too small, list:", heap_weight_node_pair)
        distributions = [(1.0 - _alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)

    nbr = [x[1] for x in heap_weight_node_pair]
    return distributions + [_alpha], nbr + [node]

def source_target_shortest_path(Gk, source, target):
    """Compute pairwise shortest path from `source` to `target` by BidirectionalDijkstra via Networkit.
    Parameters
    ----------
    source : int
        Source node index in Networkit graph `_Gk`.
    target : int
        Target node index in Networkit graph `_Gk`.
    Returns
    -------
    length : float
        Pairwise shortest path length.
    """

    length = nk.distance.BidirectionalDijkstra(Gk, source, target).run().getDistance()
    assert length < 1e300, "Shortest path between %d, %d is not found" % (source, target)
    return length

def optimal_transportation_distance(x, y, d):
    """Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY.
    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.
    Returns
    -------
    m : float
        Optimal transportation distance.
    """

    t0 = time.time()
    rho = cvx.Variable((len(y), len(x)))  # the transportation plan rho

    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)

    # m = prob.solve(solver=cvx.ECOS)  # change solver here if you want
    m = prob.solve(solver=cvx.CLARABEL)
    # solve for optimal transportation cost

#     logger.debug("%8f secs for cvxpy. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m

def xyd(Gk, source, target):
    
    x,source_topknbr = get_single_node_neighbors_distributions(Gk, source, direction="successors")
    y,target_topknbr = get_single_node_neighbors_distributions(Gk, target, direction="successors")
    
    d = []
    for src in source_topknbr:
        tmp = []
        for tgt in target_topknbr:
            tmp.append(source_target_shortest_path(Gk, src, tgt))
    d.append(tmp)
    d = np.array(d)
    
    
    return x, y, d


def G_optimal_path_w(G1, path):
    Gk = nk.nxadapter.nx2nk(G1, weightAttr='weight')

    # e0, e1: edge ; c0, c1: connection of them (neighbor)
    # find path difference e0-e1 from e0-c0-e1 or e0-c1-e1
    e0 = path['e0']
    e1 = path['e1']
    c0 = path['c0']
    c1 = path['c1']

    x_e0_e1, y_e0_e1, d_e0_e1 = xyd(Gk, e0, e1)

    path_e0_e1 = optimal_transportation_distance(np.array([x_e0_e1]).T,
                                                 np.array([y_e0_e1]).T, d_e0_e1)

    x_e0_c0, y_e0_c0, d_e0_c0 = xyd(Gk, e0, c0)

    path_e0_c0 = optimal_transportation_distance(np.array([x_e0_c0]).T,
                                                 np.array([y_e0_c0]).T, d_e0_c0)

    x_c0_e1, y_c0_e1, d_c0_e1 = xyd(Gk, c0, e1)

    path_c0_e1 = optimal_transportation_distance(np.array([x_c0_e1]).T,
                                                 np.array([y_c0_e1]).T, d_c0_e1)

    x_e0_c1, y_e0_c1, d_e0_c1 = xyd(Gk, e0, c1)

    path_e0_c1 = optimal_transportation_distance(np.array([x_e0_c1]).T,
                                                 np.array([y_e0_c1]).T, d_e0_c1)

    x_c1_e1, y_c1_e1, d_c1_e1 = xyd(Gk, c1, e1)

    path_c1_e1 = optimal_transportation_distance(np.array([x_c1_e1]).T,
                                                 np.array([y_c1_e1]).T, d_c1_e1)

    return path_e0_e1, path_e0_c0 + path_c0_e1, path_e0_c1 + path_c1_e1

def G_optimal_path(G, path):
    
    Gk = nk.nxadapter.nx2nk(G) 
    
    # e0, e1: edge ; c0, c1: connection of them (neighbor)
    # find path difference e0-e1 from e0-c0-e1 or e0-c1-e1
    e0 = path['e0']
    e1 = path['e1']
    c0 = path['c0']
    c1 = path['c1']
    
    x_e0_e1 = xyd(Gk, e0, e1)[0]
    y_e0_e1 = xyd(Gk, e0, e1)[1]
    d_e0_e1 = xyd(Gk, e0, e1)[2]
    
    path_e0_e1 = optimal_transportation_distance(np.array([x_e0_e1]).T, 
                                                 np.array([y_e0_e1]).T, d_e0_e1)
    
    x_e0_c0 = xyd(Gk, e0, c0)[0]
    y_e0_c0 = xyd(Gk, e0, c0)[1]
    d_e0_c0 = xyd(Gk, e0, c0)[2]
    
    path_e0_c0 = optimal_transportation_distance(np.array([x_e0_c0]).T, 
                                                 np.array([y_e0_c0]).T, d_e0_c0)
    
    x_c0_e1 = xyd(Gk, c0, e1)[0]
    y_c0_e1 = xyd(Gk, c0, e1)[1]
    d_c0_e1 = xyd(Gk, c0, e1)[2]
    
    path_c0_e1 = optimal_transportation_distance(np.array([x_c0_e1]).T, 
                                                 np.array([y_c0_e1]).T, d_c0_e1)

    x_e0_c1 = xyd(Gk, e0, c1)[0]
    y_e0_c1 = xyd(Gk, e0, c1)[1]
    d_e0_c1 = xyd(Gk, e0, c1)[2]
    
    path_e0_c1 = optimal_transportation_distance(np.array([x_e0_c1]).T, 
                                                 np.array([y_e0_c1]).T, d_e0_c1)
    
    x_c1_e1 = xyd(Gk, c1, e1)[0]
    y_c1_e1 = xyd(Gk, c1, e1)[1]
    d_c1_e1 = xyd(Gk, c1, e1)[2]
    
    path_c1_e1 = optimal_transportation_distance(np.array([x_c1_e1]).T, 
                                                 np.array([y_c1_e1]).T, d_c1_e1)

    return path_e0_e1, path_e0_c0+path_c0_e1, path_e0_c1+path_c1_e1

def G_shortest_path(G, path):
    
    Gk = nk.nxadapter.nx2nk(G) 
    
    # e0, e1: edge ; c0, c1: connection of them (neighbor)
    # find path difference e0-e1 from e0-c0-e1 or e0-c1-e1
    e0 = path['e0']
    e1 = path['e1']
    c0 = path['c0']
    c1 = path['c1']
    
    spath_e0_e1 = source_target_shortest_path(Gk, e0, e1)
    spath_e0_c0 = source_target_shortest_path(Gk, e0, c0)
    spath_c0_e1 = source_target_shortest_path(Gk, c0, e1)
    spath_e0_c1 = source_target_shortest_path(Gk, e0, c1)
    spath_c1_e1 = source_target_shortest_path(Gk, c1, e0)
    
    return spath_e0_e1, spath_e0_c0+spath_c0_e1, spath_e0_c1+spath_c1_e1

def G_shortest_path_w(G1, path):
    Gk = nk.nxadapter.nx2nk(G1, weightAttr='weight')

    # e0, e1: edge ; c0, c1: connection of them (neighbor)
    # find path difference e0-e1 from e0-c0-e1 or e0-c1-e1
    e0 = path['e0']
    e1 = path['e1']
    c0 = path['c0']
    c1 = path['c1']

    spath_e0_e1 = source_target_shortest_path(Gk, e0, e1)
    spath_e0_c0 = source_target_shortest_path(Gk, e0, c0)
    spath_c0_e1 = source_target_shortest_path(Gk, c0, e1)
    spath_e0_c1 = source_target_shortest_path(Gk, e0, c1)
    spath_c1_e1 = source_target_shortest_path(Gk, c1, e0)

    return spath_e0_e1, spath_e0_c0 + spath_c0_e1, spath_e0_c1 + spath_c1_e1

def path_pd_all_w(G1, path):

    path_new = copy.deepcopy(path)

    for ii in range(len(path_new)):

        op_edge = []
        op_c0 = []
        op_c1 = []
        op_c0_c1 = []

        sp_edge = []
        sp_c0 = []
        sp_c1 = []
        sp_c0_c1 = []

        for jj in range(len(path_new[ii])):
            op_edge_, op_c0_, op_c1_ = G_optimal_path_w(G1, path_new[ii].iloc[jj, :])
            op_edge.append(op_edge_)
            op_c0.append(op_c0_)
            op_c1.append(op_c1_)
            op_c0_c1.append(np.abs(op_c0_ - op_c1_))

            sp_edge_, sp_c0_, sp_c1_ = G_shortest_path_w(G1, path_new[ii].iloc[jj, :])
            sp_edge.append(sp_edge_)
            sp_c0.append(sp_c0_)
            sp_c1.append(sp_c1_)
            sp_c0_c1.append(np.abs(sp_c0_ - sp_c1_))

        path_new[ii]['op_edge'] = op_edge
        path_new[ii]['op_c0'] = op_c0
        path_new[ii]['op_c1'] = op_c1
        path_new[ii]['op_c0-c1'] = op_c0_c1


        path_new[ii]['sp_edge'] = sp_edge
        path_new[ii]['sp_c0'] = sp_c0
        path_new[ii]['sp_c1'] = sp_c1
        path_new[ii]['sp_c0-c1'] = sp_c0_c1


    return path_new

def path_pd_all(G, path):
    
    path_new = []
    
    for ii in range(len(path)):
        path[ii].insert(4, 'op_edge', 'nan')
        path[ii].insert(5, 'op_c0', 'nan')
        path[ii].insert(6, 'op_c1', 'nan')
        path[ii].insert(7, 'op_c0-c1', 'nan')
    
        path[ii].insert(8, 'sp_edge', 'nan')
        path[ii].insert(9, 'sp_c0', 'nan')
        path[ii].insert(10, 'sp_c1', 'nan')
        path[ii].insert(11, 'sp_c0-c1', 'nan')
        
        for jj in range(len(path[ii])):
            op_edge, op_c0, op_c1 = G_optimal_path(G, path[ii].iloc[jj,:])
            path[ii]['op_edge'].iloc[jj] = op_edge
            path[ii]['op_c0'].iloc[jj] = op_c0
            path[ii]['op_c1'].iloc[jj] = op_c1
            path[ii]['op_c0-c1'].iloc[jj] = np.abs(op_c0-op_c1)
        
            sp_edge, sp_c0, sp_c1 = G_shortest_path(G, path[ii].iloc[jj,:])
            path[ii]['sp_edge'].iloc[jj] = sp_edge
            path[ii]['sp_c0'].iloc[jj] = sp_c0
            path[ii]['sp_c1'].iloc[jj] = sp_c1
            path[ii]['sp_c0-c1'].iloc[jj] = np.abs(sp_c0-sp_c1)
            
        path_new.append(path)
        
    return path_new

def edge_node_vec(xyz, G, orn, ore):
    pos = {ii: (xyz.iloc[:,1:].to_numpy()[ii]) for ii in range(len(xyz))}
    
    edge_vec = pd.DataFrame(np.zeros((len(G.edges), 7)), 
                      columns=['edge', 'pos_0', 'pos_1', 'direction','ang', 
                               'distance', 'val'])
    edge_vec_pd = edge_vec.astype(object)
    
    e = list(G.edges())
    for ii in range(len(e)):
        edge_vec_pd['edge'].iloc[ii] = e[ii]
        edge_vec_pd['pos_0'].iloc[ii] = np.round(pos[e[ii][0]], 3)#.astype('float16')
        edge_vec_pd['pos_1'].iloc[ii] = np.round(pos[e[ii][1]],3)#.astype('float16')
        edge_vec_pd['direction'].iloc[ii] = np.round(edge_vec_pd['pos_1'][ii]-edge_vec_pd['pos_0'][ii],3)
        edge_vec_pd['ang'].iloc[ii] = 180*angle(np.round(pos[e[ii][0]], 3), np.round(pos[e[ii][1]],3), 'True')/np.pi
        edge_vec_pd['distance'].iloc[ii] =np.round(euclidean(pos[e[ii][1]], pos[e[ii][0]]),3)
        edge_vec_pd['val'].iloc[ii] = np.round(ore[ii],3)
        
    node_vec = pd.DataFrame(np.zeros((len(G.nodes), 8)), columns=['node', 'direction', 'x_ang', 'y_ang', 'z_ang',
                                                                  'distance', 'val', 'degree'])
    node_vec_pd = node_vec.astype(object)
    
    c = nx.barycenter(G)
    center_node=c
    
    cc = [0, 0, 0]
    for ii in range(len(c)):
        cc = np.vstack([cc, pos[c[ii]]])
        
    center = np.sum(cc, axis=0)/len(c)
    
    n = list(G.nodes())
    
    for ii in range(len(n)):

        node_vec_pd['node'].iloc[ii] = n[ii]
        node_vec_pd['direction'].iloc[ii] = np.round(pos[n[ii]]-center,3)
        node_vec_pd['x_ang'].iloc[ii] = 180*angle([1,0,0],np.round(pos[n[ii]]-center,3), 'True')/np.pi
        node_vec_pd['y_ang'].iloc[ii] = 180*angle([0,1,0],np.round(pos[n[ii]]-center,3), 'True')/np.pi
        node_vec_pd['z_ang'].iloc[ii] = 180*angle([0,0,1],np.round(pos[n[ii]]-center,3), 'True')/np.pi
        node_vec_pd['distance'].iloc[ii] = np.round(euclidean(center, pos[n[ii]]),3)
        node_vec_pd['val'].iloc[ii] = np.round(orn[n[ii]],3)
        node_vec_pd['degree'].iloc[ii] = G.degree[ii]
        
    
    return edge_vec_pd, node_vec_pd, center_node, center, pos
    
def node_based_torsion(edge_path, G):
    
    edge_list = [[] for i in range(len(G.nodes()))]
    e = np.asarray(list(G.edges()))

    for ii in range(len(G.nodes())):
        for jj in range(len(e)):
            if np.sum(np.isin(e[jj], ii))>0:
                edge_list[ii].append(jj)
            
    op_sum_node = []
    for ii in range(len(edge_list)):
        node_sum = np.sum(edge_path['op_sum'].iloc[edge_list[ii]])
        op_sum_node.append(node_sum)
    
    sp_sum_node = []
    for ii in range(len(edge_list)):
        node_sum = np.sum(edge_path['sp_sum'].iloc[edge_list[ii]])
        sp_sum_node.append(node_sum)
        
    op_mean_node = []
    for ii in range(len(edge_list)):
        node_sum = np.sum(edge_path['op_mean'].iloc[edge_list[ii]])
        op_mean_node.append(node_sum)
    
    sp_mean_node = []
    for ii in range(len(edge_list)):
        node_sum = np.sum(edge_path['sp_mean'].iloc[edge_list[ii]])
        sp_mean_node.append(node_sum)    
        
        
    return op_sum_node, sp_sum_node, op_mean_node, sp_mean_node


def edge_node_vec_torsion(xyz, G, edge_path):

    pos = {ii: (xyz.iloc[:,1:].to_numpy()[ii]) for ii in range(len(xyz))}

    op_sum_node, sp_sum_node, op_mean_node, sp_mean_node = node_based_torsion(edge_path, G)

    edge_vec = pd.DataFrame(np.zeros((len(G.edges), 10)),
                      columns=['edge', 'pos_0', 'pos_1', 'direction','ang',
                               'distance', 'op_sum', 'sp_sum', 'op_mean', 'sp_mean'])

    edge_vec_pd = edge_vec.astype(object)

    e = list(G.edges())

    for ii in range(len(e)):
        edge_vec_pd['edge'].iloc[ii] = e[ii]
        edge_vec_pd['pos_0'].iloc[ii] = np.round(pos[e[ii][0]], 3)#.astype('float16')
        edge_vec_pd['pos_1'].iloc[ii] = np.round(pos[e[ii][1]],3)#.astype('float16')
        edge_vec_pd['direction'].iloc[ii] = np.round(edge_vec_pd['pos_1'][ii]-edge_vec_pd['pos_0'][ii],3)
        edge_vec_pd['ang'].iloc[ii] = 180*angle(np.round(pos[e[ii][0]], 3), np.round(pos[e[ii][1]],3), 'True')/np.pi
        edge_vec_pd['distance'].iloc[ii] =np.round(euclidean(pos[e[ii][1]], pos[e[ii][0]]),3)
        edge_vec_pd['op_sum'].iloc[ii] = np.round(edge_path['op_sum'].iloc[ii],3)
        edge_vec_pd['sp_sum'].iloc[ii] = np.round(edge_path['sp_sum'].iloc[ii],3)
        edge_vec_pd['op_mean'].iloc[ii] = np.round(edge_path['op_mean'].iloc[ii],3)
        edge_vec_pd['sp_mean'].iloc[ii] = np.round(edge_path['sp_mean'].iloc[ii],3)

    node_vec = pd.DataFrame(np.zeros((len(G.nodes), 11)), columns=['node', 'direction', 'x_ang', 'y_ang', 'z_ang',
                                                                  'distance', 'op_sum','sp_sum',
                                                                  'op_mean', 'sp_mean','degree'])
    node_vec_pd = node_vec.astype(object)

    c = nx.barycenter(G)
    center_node=c

    cc = [0, 0, 0]
    for ii in range(len(c)):
        cc = np.vstack([cc, pos[c[ii]]])

    center = np.sum(cc, axis=0)/len(c)

    n = list(G.nodes())

    for ii in range(len(n)):

        node_vec_pd['node'].iloc[ii] = n[ii]
        node_vec_pd['direction'].iloc[ii] = np.round(pos[n[ii]]-center,3)
        node_vec_pd['x_ang'].iloc[ii] = 180*angle([1,0,0],np.round(pos[n[ii]]-center,3), 'True')/np.pi
        node_vec_pd['y_ang'].iloc[ii] = 180*angle([0,1,0],np.round(pos[n[ii]]-center,3), 'True')/np.pi
        node_vec_pd['z_ang'].iloc[ii] = 180*angle([0,0,1],np.round(pos[n[ii]]-center,3), 'True')/np.pi
        node_vec_pd['distance'].iloc[ii] = np.round(euclidean(center, pos[n[ii]]),3)
        node_vec_pd['op_sum'].iloc[ii] = np.round(op_sum_node[ii],3)
        node_vec_pd['sp_sum'].iloc[ii] = np.round(sp_sum_node[ii],3)
        node_vec_pd['op_mean'].iloc[ii] = np.round(op_mean_node[ii],3)
        node_vec_pd['sp_mean'].iloc[ii] = np.round(sp_mean_node[ii],3)
        node_vec_pd['degree'].iloc[ii] = G.degree[ii]


    return edge_vec_pd, node_vec_pd, center_node, center, pos

def angle(v1, v2, acute):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        return angle
    else:
        return 2 * np.pi - angle
    
def mirror_point(a, b, c, d, x1, y1, z1):
      
    k =(-a*x1-b*y1-c*z1-d)/float((a*a+b*b+c*c))
    x2 = a*k+x1
    y2 = b*k+y1
    z2 = c*k+z1
    x3 = 2*x2-x1
    y3 = 2*y2-y1
    z3 = 2*z2-z1
    
    return x3, y3, z3

def find_pair_node_chiral(xyz, G, orn, ore):
    
    edge_vec, node_vec, center_node, center, pos = edge_node_vec(xyz, G, orn, ore)
    
    edge_list = [[] for i in range(len(node_vec))]

    for ii in range(len(node_vec)):
        for jj in range(len(edge_vec)):
            if np.sum(np.isin(edge_vec['edge'].iloc[jj], node_vec['node'].iloc[ii]))>0:
                edge_list[ii].append(jj)
                
    edge_list_d = [[] for i in range(len(edge_list))]

    for ii in range(len(edge_list)):
        dist_u = np.unique(edge_vec.iloc[edge_list[ii]]['distance'])
        for jj in range(len(dist_u)):
            ind = np.where(np.isclose(edge_vec.iloc[edge_list[ii]]['distance'].to_numpy().astype('float64'), dist_u[jj], atol=0.1)==True)[0]
            if len(ind)>0:
                edge_list_d[ii].append(edge_vec['edge'].iloc[edge_list[ii]].iloc[ind]) 
    
    edge_list_dv = [[] for i in range(len(edge_list))]
    
    for ii in range(len(edge_list)):
        for jj in range(len(edge_list_d[ii])):
            edge_vec_list = edge_vec.iloc[edge_list[ii]]
            if len(edge_list_d[ii][jj]) > 1 :
            
                val0 = edge_vec_list['val'].iloc[np.where(edge_vec_list['edge']==np.asarray(edge_list_d[ii][jj])[0])[0][0]]
                val1 = edge_vec_list['val'].iloc[np.where(edge_vec_list['edge']==np.asarray(edge_list_d[ii][jj])[1])[0][0]]
        
                if np.isclose(val0, val1, atol=0.1)==True: # and val0*val1 >= 0:
                    edge_list_dv[ii].append(edge_list_d[ii][jj])

    
    edge_list_dv_center = edge_list_dv[center_node[0]]
    z = np.concatenate(edge_list_dv_center)
    y = [a for b in z for a in b]
    a = collections.Counter(y)
    drop = np.where(np.array(list(a.items()))[:,1]>1)[0][0]
    node_id = np.delete(np.array(list(a.items()))[:,0], drop)
    
    ang = np.hstack([node_vec['x_ang'].to_numpy().reshape(-1,1),
                    node_vec['y_ang'].to_numpy().reshape(-1,1),
                    node_vec['z_ang'].to_numpy().reshape(-1,1)])
    
    ang_sum = []
    comb = list(itertools.combinations(node_id,2))
    for ii in range(len(comb)):
        s = np.sum(np.partition(np.abs(ang[comb[ii][0]]-ang[comb[ii][1]]),1)[0:2])
        ang_sum.append(s)
    
    pair = []    
    
    for ii in range(len(ang_sum)):
        for jj in range(len(np.unique(ang_sum))):
            p = np.array(comb)[np.where(ang_sum==np.unique(ang_sum)[jj])[0]]
            if p.size != 0:
                pair.append(p)

    pair_new_ = [i[np.sort(np.unique(i, axis=0, return_index=True)[1])][0] for i in pair]
    pair_new = np.array(pair_new_)

    inter1 = []
    inter1.append(pair_new[0])
    for ii in range(len(pair_new)):
        if len(np.intersect1d(pair_new[ii], pair_new[0])) ==0:
            inter1.append(pair_new[ii])
            
    inter2 = []
    inter2.append(inter1[0])
    inter2.append(inter1[1])
    for ii in range(2, len(inter1)):
        if len(np.intersect1d(inter1[ii], inter1[1]))==0:
            inter2.append(inter1[ii])
        
    inter = inter2[:3]

    return inter
                
def find_symmetric_point_and_plane_chiral(xyz, G, orn, ore):
       
    edge_vec, node_vec, center_node, center, pos = edge_node_vec(xyz, G, orn, ore)
    node_pair = find_pair_node_chiral(xyz, G, orn, ore)
    final_pair = pd.DataFrame(node_pair).drop_duplicates().to_numpy()[:3]
    
    points = []
    
    for ii in range(len(final_pair)):
        n0 = final_pair[ii][0]
        n1 = final_pair[ii][1]
        p = (node_vec['direction'].iloc[n0]+node_vec['direction'].iloc[n1])/2
        points.append(p)
            
    v1 = points[1]-points[0]
    v2 = points[2]-points[0]

    cp = np.cross(v1, v2)

    a, b, c = cp
    d = np.dot(cp, points[0])
    
    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    
                
    return a,b,c,d

def mirror_check_chiral(xyz, G, orn, ore):
    
    edge_vec, node_vec, center_node, center, pos = edge_node_vec(xyz, G, orn, ore)
    node_pair  = find_pair_node_chiral(xyz, G, orn, ore)
    a,b,c,d = find_symmetric_point_and_plane_chiral(xyz, G, orn, ore)
    
    m_point = []

    for ii in range(len(node_vec['direction'])):
        point = mirror_point(a,b,c,-d, node_vec['direction'].iloc[ii][0],
                node_vec['direction'].iloc[ii][1],
                node_vec['direction'].iloc[ii][2])
        m_point.append(point)
        
    pair_ind = []
    for ii in range(len(m_point)):
        S = np.sum(np.isclose(np.vstack(node_vec['direction']),np.round(np.asarray(m_point)[ii],3), atol=0.3), axis=1)
        if len(np.where(S==3)[0]) != 0:
            pair_ind.append(np.where(S==3)[0])
        else:
            pair_ind.append(np.nan)

            
    mirror_pair = np.hstack([np.arange(len(m_point)).reshape(-1,1), np.vstack(pair_ind)])
            
    return mirror_pair, a, b, c, d


def chirality_sign(xyz, G, edge_path, orn, ore):
    edge_vec, node_vec, center_node, center, pos = edge_node_vec_torsion(xyz, G, edge_path)
    node_pair, a,b,c,d = mirror_check_chiral(xyz, G, orn, ore)

    plane_sign = pd.DataFrame(np.zeros((len(node_pair), 4)), columns=['n0', 'n0_sign',
                                                                      'n1', 'n1_sign', ])

    plane_sign['n0'] = node_pair[:, 0]
    plane_sign['n1'] = node_pair[:, 1]

    for ii in range(len(plane_sign)):
        n0 = plane_sign['n0'].iloc[ii].astype('int32')
        n0_x = node_vec['direction'].iloc[n0][0]
        n0_y = node_vec['direction'].iloc[n0][1]
        n0_z = node_vec['direction'].iloc[n0][2]

        plane_sign['n0_sign'].iloc[ii] = np.sign(a * n0_x + b * n0_y + c * n0_z - d)

    for ii in range(len(plane_sign)):
        if np.isnan(plane_sign['n1'].iloc[ii]) == False:
            n1 = plane_sign['n1'].iloc[ii].astype('int32')
            n1_x = node_vec['direction'].iloc[n1][0]
            n1_y = node_vec['direction'].iloc[n1][1]
            n1_z = node_vec['direction'].iloc[n1][2]

            plane_sign['n1_sign'].iloc[ii] = np.sign(a * n1_x + b * n1_y + c * n1_z - d)

    plus_diff = []
    plus_val_diff = []

    p_ind = np.where(plane_sign['n0_sign'] > 0)[0]

    for ii in range(len(p_ind)):
        if np.isnan(plane_sign['n1'].iloc[p_ind[ii]]) == True:

            n0 = node_pair[p_ind[ii]][0].astype('int32')
            m_point = mirror_point(a, b, c, -d, node_vec['direction'].iloc[n0][0],
                                   node_vec['direction'].iloc[n0][1],
                                   node_vec['direction'].iloc[n0][2])
            m_point_arr = np.asarray(m_point)
            diff = euclidean(m_point_arr, [0, 0, 0])
            plus_diff.append(diff)

            val = node_vec['op_mean'].iloc[n0]
            plus_val_diff.append(np.abs(val))

        else:
            n0 = node_pair[p_ind[ii]][0].astype('int32')
            n1 = node_pair[p_ind[ii]][1].astype('int32')
            m_point = mirror_point(a, b, c, -d, node_vec['direction'].iloc[n0][0],
                                   node_vec['direction'].iloc[n0][1],
                                   node_vec['direction'].iloc[n0][2])
            m_point_arr = np.asarray(m_point)
            diff = euclidean(node_vec['direction'].iloc[n1], m_point_arr)
            plus_diff.append(diff)

            val0 = node_vec['op_mean'].iloc[n0]
            val1 = node_vec['op_mean'].iloc[n1]

            plus_val_diff.append(np.abs(val0 - val1))

    minus_diff = []
    minus_val_diff = []

    m_ind = np.where(plane_sign['n0_sign'] < 0)[0]

    for ii in range(len(m_ind)):
        if np.isnan(plane_sign['n1'].iloc[m_ind[ii]]) == True:

            n0 = node_pair[m_ind[ii]][0].astype('int32')
            m_point = mirror_point(a, b, c, -d, node_vec['direction'].iloc[n0][0],
                                   node_vec['direction'].iloc[n0][1],
                                   node_vec['direction'].iloc[n0][2])
            m_point_arr = np.asarray(m_point)
            diff = euclidean(m_point_arr, [0, 0, 0])
            minus_diff.append(diff)

            val = node_vec['op_mean'].iloc[n0]
            minus_val_diff.append(np.abs(val))

        else:
            n0 = node_pair[m_ind[ii]][0].astype('int32')
            n1 = node_pair[m_ind[ii]][1].astype('int32')
            m_point = mirror_point(a, b, c, -d, node_vec['direction'].iloc[n0][0],
                                   node_vec['direction'].iloc[n0][1],
                                   node_vec['direction'].iloc[n0][2])
            m_point_arr = np.asarray(m_point)
            diff = euclidean(node_vec['direction'].iloc[n1], m_point_arr)
            minus_diff.append(diff)

            val0 = node_vec['op_mean'].iloc[n0]
            val1 = node_vec['op_mean'].iloc[n1]

            minus_val_diff.append(np.abs(val0 - val1))

    chirality = np.sum(np.asarray(plus_diff) * np.asarray(plus_val_diff)) / len(plus_diff) - np.sum(
        np.asarray(minus_diff) * np.asarray(minus_val_diff)) / len(minus_diff)

    return chirality