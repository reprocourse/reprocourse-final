""" reprocourse

NEUR602 class project paper coding, reproducing a paper

1. [x] Rich club result
2. [x] contrast proportions of density and communication cost (Fig. 2/3)
3. [x] path patterns (local motifs) (Fig. 3) (optional)
4. [o] path and dependence on node degree (Fig. 4) (optional)

References
Methods:    Heuvel et al. 2012

Project:    networks-1
File Name:  script_20181116_submit
Author:     Zhen-Qi Liu
Date Created:   16/11/18
Last Modified:  02/12/18
Code Status:    FINAL
"""

from tqdm import tqdm
import numpy as np
import brainconn as bc
import networkx as nx
from itertools import product, groupby
from functools import partial
from collections import Counter
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from nilearn import plotting as plotting


def read_mat(path):
    """

    Support function to read .mat data

    :param path:
    :return:
    """
    import scipy.io
    mat = scipy.io.loadmat(path)
    return mat


def save_pickle_file(path, data):
    """

    Support function to save pickle file

    :param path:
    :param data:
    :return:
    """
    import pickle
    with open(f'{path:}.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=-1)


def load_pickle_file(path):
    """

    Support function to load pickle file

    :param path:
    :return:
    """
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def run_mp(part_func, param_list):
    """

    Support function to run multiprocessing locally

    :param part_func:
    use partial to generate a partial function with only one parameter
    > from functools import partial
    > foo_p = partial(foo, all_params_except_one=default)
    :param param_list:
    list of variable parameter in the partial function
    :return: results
    if labelled result is desired, return a tuple containing the input
    """
    import gc
    import multiprocessing as mp
    gc.collect()
    pool = mp.Pool(int(mp.cpu_count()))
    ret = pool.map(part_func, param_list)
    pool.close()
    return ret


def load_heuvel_data():
    """

    Load principle and replication dataset for this project

    :return:
    """
    mat = read_mat('../Original-Data/GR_Dataset_n2x40.mat')
    conn_gr1, conn_gr2 = np.array(mat['GROUP_MATRIX_HD_gr1']), np.array(mat['GROUP_MATRIX_HD_gr2'])
    dist_gr1, dist_gr2 = np.array(mat['GROUP_MATRIX_HDlength_gr1']), np.array(mat['GROUP_MATRIX_HDlength_gr2'])
    return conn_gr1, conn_gr2, dist_gr1, dist_gr2


def prepare_g_conn(conn_mat, dist_mat):
    g_conn = nx.from_numpy_matrix(conn_mat, create_using=nx.Graph())
    conn_mat_bd = bc.utils.binarize(conn_mat, copy=True)
    for it1 in range(conn_mat.shape[0]):
        for it2 in range(dist_mat.shape[1]):
            if (it1, it2) in g_conn.edges:
                g_conn[it1][it2]['distance'] = dist_mat[it1, it2]
    return g_conn


def rich_club_detection():
    redo_rich_club_simu = 1
    if redo_rich_club_simu:
        conn_rc = bc.core.rich_club_wu(conn_mat, klevel=100)
        #
        randmat_count = 100
        randmat_rc = np.zeros((randmat_count, conn_rc.shape[0]))
        for i in tqdm(range(randmat_count)):
            randmat, _ = bc.reference.randmio_und(conn_mat, 50)
            tmp_rc = bc.core.rich_club_wu(randmat, klevel=100)
            randmat_rc[i, :] = tmp_rc
        randmat_rc_mean = np.mean(randmat_rc, axis=0)
        conn_rc_ratio = conn_rc / randmat_rc_mean
        #
        randmat_rc = np.nan_to_num(randmat_rc)
        conn_rc = np.nan_to_num(conn_rc)
        p_vals = np.sum(randmat_rc >= conn_rc, axis=0) / randmat_count
        p_sig_index = np.where(p_vals < 0.05)
        #
        save_pickle_file(f'../Analysis-Data/rc',
                         [conn_rc, randmat_rc, randmat_rc_mean, conn_rc_ratio, p_vals, p_sig_index])


def rich_club_plot():
    [conn_rc, randmat_rc, randmat_rc_mean,
     conn_rc_ratio, p_vals, p_sig_index] = load_pickle_file('../Analysis-Data/rc.pickle')
    # rich club significance graph
    plt.figure()
    plt.plot(conn_rc_ratio, 'bo-')
    plt.plot(p_sig_index[0], conn_rc_ratio[p_sig_index], 'ro-')
    plt.xlabel('node degree')
    plt.ylabel('rich club ratio')
    plt.title('Rich club analysis for the network')
    plt.show()


def get_rc_components(g_conn):
    # find rich club components
    # networkx degree calc
    g_conn_degrees = list(nx.degree(g_conn))
    # get rc nodes
    rich_club_thres = 10
    g_conn_rc_nodes = [_[0] for _ in g_conn_degrees if _[1] > rich_club_thres]
    g_conn_non_rc_nodes = [_ for _ in list(g_conn) if _ not in g_conn_rc_nodes]
    # get rc edges
    g_conn_edges = list(nx.edges(g_conn))
    g_conn_rc_edges = [_ for _ in g_conn_edges if _[0] in g_conn_rc_nodes and _[1] in g_conn_rc_nodes]
    g_conn_local_edges = [_ for _ in g_conn_edges if _[0] in g_conn_non_rc_nodes and _[1] in g_conn_non_rc_nodes]
    g_conn_feeder_edges = [_ for _ in g_conn_edges if _ not in g_conn_rc_edges and _ not in g_conn_local_edges]
    rc_types = [g_conn_rc_edges, g_conn_local_edges, g_conn_feeder_edges]

    return {'g_conn': g_conn,
            'g_conn_rc_nodes': g_conn_rc_nodes,
            'g_conn_non_rc_nodes': g_conn_non_rc_nodes,
            'g_conn_edges': g_conn_edges,
            'g_conn_rc_edges': g_conn_rc_edges,
            'g_conn_local_edges': g_conn_local_edges,
            'g_conn_feeder_edges': g_conn_feeder_edges,
            'rc_types': rc_types
            }


def trade_off_1(rc_comp):
    g_conn, g_conn_edges, rc_types = rc_comp['g_conn'], rc_comp['g_conn_edges'], rc_comp['rc_types']
    [g_conn_rc_edges, g_conn_local_edges, g_conn_feeder_edges] = rc_types

    curr_edges = [_ for _ in g_conn_edges]
    rc_assign = []
    for edge in curr_edges:
        if edge in g_conn_rc_edges:
            rc_assign.append('3 rich_club')
        elif edge in g_conn_local_edges:
            rc_assign.append('1 local')
        elif edge in g_conn_feeder_edges:
            rc_assign.append('2 feeder')
        else:
            rc_assign.append('error')
    #
    dict_prop_1 = {
        'edges': curr_edges,
        'dist': [g_conn.get_edge_data(*_)['distance'] for _ in curr_edges],
        'dist*weight': [g_conn.get_edge_data(*_)['distance'] * g_conn.get_edge_data(*_)['weight'] for _ in curr_edges],
        'rc_type': rc_assign
    }
    df_prop_1 = pd.DataFrame(data=dict_prop_1)
    df_prop_1['dist_type'] = pd.cut(df_prop_1['dist'], [-1, 29.999, 90.001, 1000], labels=['low', 'mid', 'high'])
    # df_prop_1['dist_type'] = pd.qcut(df_prop_1['dist*weight'], 3, labels=['low', 'mid', 'high'])
    # df_prop_1.groupby(['dist_type', 'rc_type'])['edges'].count().unstack().plot(kind='bar', stacked=True)

    # Figure 2A
    print(
        df_prop_1 \
            .groupby(['dist_type', 'rc_type']) \
            .agg({'edges': 'count'}) \
            .groupby(level=0) \
            .apply(lambda x: 100 * x / float(x.sum()))
    )


def get_shortest_path_wrapper(pos, hops, Pmat):
    path_tmp = bc.distance.retrieve_shortest_path(pos[0], pos[1], hops, Pmat)
    return (pos[0], pos[1], [_[0] for _ in path_tmp])


def get_edge_tuple_from_node_list(nodes):
    return [(nodes[idx], nodes[idx + 1]) for idx, _ in enumerate(nodes[:-1])]


def transform_edge_list_to_type(edge_list, rc_comp):
    types = rc_comp['rc_types']
    [g_conn_rc_edges, g_conn_local_edges, g_conn_feeder_edges] = types
    expr = ''
    for edge in edge_list:
        if edge in g_conn_rc_edges or edge[::-1] in g_conn_rc_edges:
            expr += 'R'
        elif edge in g_conn_feeder_edges or edge[::-1] in g_conn_feeder_edges:
            expr += 'F'
        elif edge in g_conn_local_edges or edge[::-1] in g_conn_local_edges:
            expr += 'L'
        else:
            expr += 'N'
    return ''.join(i for i, _ in groupby(expr))


def get_edge_type_cost_from_egde_list(edge_list, rc_comp):
    g_conn = rc_comp['g_conn']
    types = rc_comp['rc_types']
    [g_conn_rc_edges, g_conn_local_edges, g_conn_feeder_edges] = types
    rc, feeder, local, other = 0, 0, 0, 0
    for edge in edge_list:
        wei, dist = g_conn.get_edge_data(edge[0], edge[1]).values()
        if edge in g_conn_rc_edges or edge[::-1] in g_conn_rc_edges:
            rc += wei * dist
        elif edge in g_conn_feeder_edges or edge[::-1] in g_conn_feeder_edges:
            feeder += wei * dist
        elif edge in g_conn_local_edges or edge[::-1] in g_conn_local_edges:
            local += wei * dist
        else:
            other += wei * dist
    return (rc, feeder, local, other)


def find_all_shortest_path_pairs(conn_mat):
    # find all shortest paths between all pair of nodes
    shortest_paths_nodes = []
    _, hops, Pmat = bc.distance.distance_wei_floyd(conn_mat)
    pos_list = [_ for _ in product(range(conn_mat.shape[0]), repeat=2) if _[0] < _[1]]
    shortest_path_partial = partial(get_shortest_path_wrapper, hops=hops, Pmat=Pmat)
    shortest_paths_nodes = run_mp(shortest_path_partial, pos_list)  # takes about 20-30s
    shortest_paths = [get_edge_tuple_from_node_list(_[2]) for _ in shortest_paths_nodes]

    save_pickle_file(f'../Analysis-Data/shortest_path_pairs',
                     [shortest_paths_nodes,
                      shortest_paths]
                     )


def get_comm_cost(g_conn, rc_comp):
    [shortest_paths_nodes, shortest_paths] = load_pickle_file('../Analysis-Data/shortest_path_pairs.pickle')
    types = rc_comp['rc_types']
    [g_conn_rc_edges, g_conn_local_edges, g_conn_feeder_edges] = types
    # communication cost
    cost_partial = partial(get_edge_type_cost_from_egde_list, rc_comp=rc_comp)
    edge_costs = run_mp(cost_partial, shortest_paths)

    save_pickle_file(f'../Analysis-Data/edge_costs',
                     [edge_costs]
                     )


def show_comm_cost(rc_comp):
    types = rc_comp['rc_types']
    g_conn_edges = rc_comp['g_conn_edges']
    [g_conn_rc_edges, g_conn_local_edges, g_conn_feeder_edges] = types
    edge_costs = load_pickle_file('../Analysis-Data/edge_costs.pickle')[0]
    print('original solution - cost')
    print(f'rc   feeder  local   other')
    print(f'{np.sum(np.array(edge_costs), axis=0)}')
    print('as percentage')
    print(f'{np.sum(np.array(edge_costs), axis=0) / np.sum(np.array(edge_costs))}')

    print('Another solution - density and cost')
    print(
        f'{np.array([len(g_conn_rc_edges), len(g_conn_feeder_edges), len(g_conn_local_edges)]) / len(g_conn_edges)}')
    rc_cost = np.sum([conn_mat[i, j] * dist_mat[i, j] for (i, j) in g_conn_rc_edges])
    local_cost = np.sum([conn_mat[i, j] * dist_mat[i, j] for (i, j) in g_conn_local_edges])
    feeder_cost = np.sum([conn_mat[i, j] * dist_mat[i, j] for (i, j) in g_conn_feeder_edges])
    print(
        f'{np.array([rc_cost, feeder_cost, local_cost]) / (rc_cost + local_cost + feeder_cost)}')


def get_all_path_exprs(rc_comp):
    [shortest_paths_nodes, shortest_paths] = load_pickle_file('../Analysis-Data/shortest_path_pairs.pickle')
    # path motifs
    transform_partial = partial(transform_edge_list_to_type, rc_comp=rc_comp)
    shortest_paths_expr = run_mp(transform_partial, shortest_paths)  # takes 10 min

    c_path_motifs = Counter(shortest_paths_expr)

    save_pickle_file(f'../Analysis-Data/path_motifs',
                     [shortest_paths_expr,
                      c_path_motifs]
                     )


def get_path_motifs():
    [shortest_paths_expr, c_path_motifs] = load_pickle_file('../Analysis-Data/path_motifs.pickle')

    c_dict = list(dict(c_path_motifs).items())
    c_sum = np.sum([_[1] for _ in c_dict])
    c_dict = [(_[0], _[1] / c_sum) for _ in c_dict if _[1] / c_sum > 0.01]

    c_dict = sorted(c_dict, key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots()
    y_pos = np.arange(len(c_dict))
    y_label, x_val = zip(*c_dict)
    ax.barh(y_pos, x_val, align='center', color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_label)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Percentage of occurrence')
    ax.set_title('Frequencies of selected path motifs')

    original_rank = ['LFRFL', 'LFL', 'LFRF', 'FRFL', 'L', 'LFR', 'RFL', 'LF',
                     'FL', 'FRF', 'LFLFL', 'LFLF', 'FLFL',
                     'LFRFLFL', 'LFRFLFL', 'FR', 'RF', 'LFRFLF', 'FLFRFL',
                     'LFLFRF', 'FRFLFL', 'FLF', 'LFLFR',
                     'RFLFL', 'FLFRF', 'FRFLF', 'F']
    current_rank = [_[0] for _ in c_dict]

    original_values = [(_, idx) for idx, _ in enumerate(original_rank)]
    current_values = [(_, current_rank.index(_)) if _ in current_rank
                      else (_, 'NaN') for _ in original_rank]

    rank_series = [(original_values[i][1], current_values[i][1]) for i in range(len(original_rank))]

    plt.figure()
    plt.gca().invert_yaxis()
    plt.xticks([0, 1], ['original', 'current'])
    plt.yticks(range(len(original_rank)), original_rank)
    for line in rank_series:
        if line[1] != 'NaN':
            if abs(line[0] - line[1]) <= 3:
                plt.plot((0, 1), line, '-bo')
            else:
                plt.plot((0, 1), line, '-ro')


if __name__ == '__main__':
    ### preparatory ###
    # prepare data
    conn_gr1, conn_gr2, dist_gr1, dist_gr2 = load_heuvel_data()
    # change this line to switch between principle and
    conn_mat, dist_mat = conn_gr1.copy(), dist_gr1.copy()
    # prepare networkx graph object
    g_conn = prepare_g_conn(conn_mat, dist_mat)
    # plot the connectivity matrix
    # plotting.plot_matrix(conn_mat, cmap=cm.get_cmap('viridis'))
    # plt.show()
    # plot the distance matrix
    # plotting.plot_matrix(dist_mat, cmap=cm.get_cmap('viridis'))
    # plt.show()

    ### rich club detection ###
    # run this function one time to get rich club data
    # rich_club_detection()
    # get the plot
    # rich_club_plot()
    # find rich club components
    rc_comp = get_rc_components(g_conn)

    ### connection types grouped by length ###
    # get the result
    # trade_off_1(rc_comp)

    ### network density and communication cost ###
    # run this function one time to get data for the next few figures
    # (takes several minutes)
    # find_all_shortest_path_pairs(conn_mat)
    # get the result
    # (takes several minutes)
    # get_comm_cost(g_conn, rc_comp)
    # show_comm_cost(rc_comp)

    ### path motifs ###
    # run this function one time to get data for the next figure
    # get_all_path_exprs(rc_comp)
    # get the result
    # get_path_motifs()
