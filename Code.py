import pandas as pd
import numpy as np
import seaborn as sns
import random
from points_io import save_points_as_pdb
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.ndimage import zoom
from scipy.stats import pearsonr

def get_data(path: str, chromosome : str = b'15') -> np.array:
    data = np.genfromtxt(path, dtype='object', skip_header=1)
    filtered = data[((data[:, 0] == chromosome) & (data[:, 2] == chromosome))]
    filtered = filtered[:, [1,3]].astype(int)
    return filtered

def create_contact_matrix(data: np.array, size: int = 1000000) -> np.matrix:
    chromosome_size = np.max(data) + 1
    num_bins = chromosome_size // size + 1
    matrix = np.zeros((num_bins, num_bins))
    for el in data:
        matrix[el[0] // size, el[1] // size] += 1
        matrix[el[1] // size, el[0] // size] += 1
    
    return matrix


def generate_self_avoiding_walk(max_steps: int, grid: int) -> list:

    walk = [(0, 0, 0)]
    visited = set([(0, 0, 0)])
    
    moves = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    
    for _ in range(max_steps):
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
    return save_points_as_pdb(array, pdb_file_name=filename)

def generate_graph(points: list) -> nx.graph:
    G = nx.Graph()

    for i, coord in enumerate(points):
        G.add_node(i, pos=coord)

    for i in range(len(points) - 1):
        G.add_edge(i, i + 1)
    return G

def __get_hic_map(points: list) -> np.array:
    matrix_of_distances = distance_matrix(points, points)
    max_distance = np.max(matrix_of_distances)
    matrix_of_distances = max_distance - matrix_of_distances
    return matrix_of_distances

def __change_dimension(matrix_of_distances: np.array, size: int) -> np.matrix:
    change_rate = size / matrix_of_distances.shape[0]
    changed_dimension_matrix = zoom(matrix_of_distances, change_rate, order=1)
    return changed_dimension_matrix

def __pearson_corelation(original_matrix, new_matrix):
    corelation, pvalue = pearsonr(original_matrix.flatten(), new_matrix.flatten())
    return corelation

def f_function(points, original_matrix):
    matrix_of_distances = __get_hic_map(points)
    changed_dimension_matrix = __change_dimension(matrix_of_distances, original_matrix.shape[0])
    corelation = __pearson_corelation(original_matrix, changed_dimension_matrix)
    print('Pearson correlation = ', corelation)
    return corelation

