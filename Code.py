import pandas as pd
import numpy as np
import seaborn as sns
import random
from points_io import save_points_as_pdb
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.ndimage import zoom
from scipy.stats import pearsonr
import math

def get_data(path: str, chromosome : str = b'15') -> np.array:
    """
    Reads and filters data from a specified file path based on the given chromosome.
    
    Parameters:
    path (str): Path to the data file.
    chromosome (str): Chromosome number to filter by (default is '15').
    
    Returns:
    np.array: Filtered data containing only the specified chromosome interactions.
    """

    data = np.genfromtxt(path, dtype='object', skip_header=1)
    filtered = data[((data[:, 0] == chromosome) & (data[:, 2] == chromosome))]
    filtered = filtered[:, [1,3]].astype(int)
    return filtered

def create_contact_matrix(data: np.array, size: int = 1000000) -> np.matrix:
    """
    Creates a contact matrix from the filtered data.
    
    Parameters:
    data (np.array): Filtered data array.
    size (int): Size of the bins to aggregate contacts (default is 1,000,000).
    
    Returns:
    np.matrix: Contact matrix representing the aggregated contacts.
    """

    chromosome_size = np.max(data) + 1
    num_bins = chromosome_size // size + 1
    matrix = np.zeros((num_bins, num_bins))
    for el in data:
        matrix[el[0] // size, el[1] // size] += 1
        matrix[el[1] // size, el[0] // size] += 1
    
    return matrix


def generate_self_avoiding_walk(max_steps: int, grid: int) -> list:
    """
    Generates a self-avoiding walk within a specified grid size.
    
    Parameters:
    max_steps (int): Maximum number of steps in the walk.
    grid (int): Grid size for the walk.
    
    Returns:
    list: List of coordinates representing the self-avoiding walk.
    """


    walk = [(0, 0, 0)]
    visited = set([(0, 0, 0)])
    
    moves = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]

    for i in range(max_steps):
        dx, dy, dz = random.choice(moves)
        new_pos = (walk[-1][0] + dx, walk[-1][1] + dy, walk[-1][2] + dz)
        
        if any(abs(i) > grid for i in new_pos):
            continue
        
        if new_pos not in visited:
            walk.append(new_pos)
            visited.add(new_pos)
        
        else:
            continue        
    return walk

def save_to_pdb(array: np.array, filename: str = 'points.pdb') -> None:
    """
    Saves an array of points to a PDB file.
    
    Parameters:
    array (np.array): Array of points to save.
    filename (str): Name of the PDB file (default is 'points.pdb').
    """

    return save_points_as_pdb(array, pdb_file_name=filename)

def generate_graph(points: list) -> nx.graph:
    """
    Generates a graph from a list of points.
    
    Parameters:
    points (list): List of coordinates to create nodes in the graph.
    
    Returns:
    nx.Graph: Graph with nodes representing the points and edges connecting consecutive points.
    """

    G = nx.Graph()

    for i, coord in enumerate(points):
        G.add_node(i, pos=coord)

    for i in range(len(points) - 1):
        G.add_edge(i, i + 1)
    return G

def __get_hic_map(points: list) -> np.array:
    """
    Computes the Hi-C map (distance matrix) from a list of points.
    
    Parameters:
    points (list): List of coordinates.
    
    Returns:
    np.array: Hi-C map represented as a distance matrix.
    """

    matrix_of_distances = distance_matrix(points, points)
    max_distance = np.max(matrix_of_distances)
    matrix_of_distances = np.abs(matrix_of_distances - max_distance)
    return matrix_of_distances

def __change_dimension(matrix_of_distances: np.array, size: int) -> np.matrix:
    """
    Changes the dimension of a distance matrix to a specified size using interpolation.
    
    Parameters:
    matrix_of_distances (np.array): Original distance matrix.
    size (int): Desired size of the new matrix.
    
    Returns:
    np.matrix: Resized distance matrix.
    """

    change_rate = size / matrix_of_distances.shape[0]
    changed_dimension_matrix = zoom(matrix_of_distances, change_rate, order=1)
    return changed_dimension_matrix

def __pearson_corelation(original_matrix: np.matrix, new_matrix: np.matrix) -> float:
    """
    Calculates the Pearson correlation coefficient between two matrices.
    
    Parameters:
    original_matrix (np.matrix): Original matrix.
    new_matrix (np.matrix): New matrix to compare with the original.
    
    Returns:
    float: Pearson correlation coefficient between the two matrices.
    """

    corelation, pvalue = pearsonr(original_matrix.flatten(), new_matrix.flatten())
    return corelation

def f_function(points: list, original_matrix: np.matrix) -> float:
    """
    Computes the correlation between the Hi-C map of a given set of points and an original matrix.
    
    Parameters:
    points (list): List of coordinates.
    original_matrix (np.matrix): Original Hi-C matrix.
    
    Returns:
    float: Pearson correlation coefficient between the Hi-C map of points and the original matrix.
    """

    matrix_of_distances = __get_hic_map(points)
    changed_dimension_matrix = __change_dimension(matrix_of_distances, original_matrix.shape[0])
    corelation = __pearson_corelation(original_matrix, changed_dimension_matrix)
    return corelation

def initalize_route(graph: nx.graph) -> list:
    """
    Initializes a route on a graph by selecting a random start node and one of its neighbors.
    
    Parameters:
    graph (nx.Graph): Graph to initialize the route on.
    
    Returns:
    list: Initial route containing the start node and its neighbor.
    """

    start = random.choice(list(graph.nodes()))
    route = [graph.nodes[start]['pos']]
    neighbour = random.choice(list(graph.neighbors(start)))
    route.append(graph.nodes[neighbour]['pos'])
    nodes = [start, neighbour]
    return route, nodes

def __add_to_route(graph: nx.graph, route: list, nodes: list) -> list:
    """
    Adds a new node to the route by randomly selecting a neighbor of a first or last node in the route.
    
    Parameters:
    graph (nx.Graph): Graph containing the nodes.
    route (list): Current route.
    nodes (list): List of nodes in the current route.
    
    Returns:
    list: Updated route and nodes list with a new node added.
    """

    wsk = - 1 if random.uniform(0,1) > 0.5 else 0
    neighbours = list(filter(lambda el: el not in nodes, list(graph.neighbors(nodes[wsk]))))
    if len(neighbours) != 0:
        neighbour = random.choice(neighbours)
        if wsk == 0:
            nodes[:0] = [neighbour]
            route[:0] = [graph.nodes[neighbour]['pos']]
        else:
            nodes.append(neighbour)
            route.append(graph.nodes[neighbour]['pos'])
    return route, nodes


def __change_edges_to_one(route) -> list:
    """
    Removes a random point from the route, reducing the number of edges by one.
    
    Parameters:
    route (list): Current route.
    
    Returns:
    list: Updated route with one point removed.
    """

    indx = random.randint(1, len(route) - 2)
    route.pop(indx)
    return route

def __change_edge_to_two(route) -> list:
    """
    Inserts a new point between two existing points in the route, increasing the number of edges by one.
    
    Parameters:
    route (list): Current route.
    
    Returns:
    list: Updated route with a new point inserted.
    """

    indx = random.randint(0, len(route) - 2)
    left = route[indx]
    diff = [random.uniform(-1/2,1/2) for _ in range(3)]
    point = [left[i] + diff[i] for i in range(3)]
    route.insert(indx + 1, point)
    return route


def g_function(graph: nx.graph, route: list, nodes: list)-> list:
    """
    Modifies the route by either adding a new node, inserting a point, or removing a point.
    
    Parameters:
    graph (nx.Graph): Graph containing the nodes.
    route (list): Current route.
    nodes (list): List of nodes in the current route.
    
    Returns:
    list: Updated route and nodes list after modification.
    """

    prob = random.uniform(0,1)
    if prob < 0.4:
        route, nodes = __add_to_route(graph,route,nodes)
    elif prob < 0.9:
        route = __change_edge_to_two(route)
    else:
        if len(route) > 2:
            route = __change_edges_to_one(route)
    return route, nodes

def __accept_func(corr, corr_prop, t) -> float:
    """
    Calculates the acceptance probability for a proposed route based on the correlation and temperature.
    
    Parameters:
    corr (float): Correlation of the current route.
    corr_prop (float): Correlation of the proposed route.
    t (float): Current temperature.
    
    Returns:
    float: Acceptance probability for the proposed route.
    """

    prob = min(math.exp(-(corr -corr_prop)/t), 1)
    return prob

def simmulated_annealing(graph, t_init, matrix, epochs=1000) -> list:
    """
    Performs simulated annealing to find a route that maximizes the correlation with the given matrix.
    
    Parameters:
    graph (nx.Graph): Graph containing the nodes.
    t_init (float): Initial temperature.
    matrix (np.matrix): Original Hi-C matrix.
    epochs (int): Number of epochs for the annealing process (default is 1000).
    
    Returns:
    list: Final route, list of correlations over epochs, acceptance rates, and steps taken.
    """

    route, nodes = initalize_route(graph)
    corr = f_function(route, matrix)
    accept = []
    steps = []
    correlations = [corr]
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('Epoch: ', epoch) 
        prob = random.uniform(0,1)
        t = t_init * (1 - epoch / epochs)
        route_prop, nodes_prop = g_function(graph, route.copy(), nodes)
        corr_prop = f_function(route_prop, matrix)
        accept_rate = __accept_func(corr, corr_prop, t)
        accept.append(accept_rate)
        if  accept_rate > prob:
            if len(route) < len(route_prop):
                steps.append(1)
            elif len(route) == len(route_prop):
                steps.append(0)
            else:
                steps.append(-1)

            route = route_prop.copy()
            nodes = nodes_prop
            corr = corr_prop
        else:
            steps.append(0)
        correlations.append(corr)
    return route, correlations, accept, steps

def accepted_move_types(steps) -> list:
    """
    Analyzes the types of moves (step forward, step stay, step back) over the simulation process.
    
    Parameters:
    steps (list): List of steps taken during the simulation.
    
    Returns:
    list: Lists representing the proportion of each move type (step forward, step stay, step back) at each epoch.
    """

    step_back = [step == -1 for step in steps]
    step_stay = [step == 0 for step in steps]
    step_forrward = [step == 1 for step in steps]
    length = len(steps)

    step_back_new = []
    step_stay_new = []
    step_forrward_new = []
    for i in range(length):
        step_back_new.append(sum(step_back[i:]) / (length - i))
        step_stay_new.append(sum(step_stay[i:]) / (length - i))
        step_forrward_new.append(sum(step_forrward[i:]) / (length - i))
    return [step_back_new, step_stay_new, step_forrward_new]


    







