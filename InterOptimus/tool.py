from pymatgen.transformations.standard_transformations import DeformStructureTransformation
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.surface import SlabGenerator
import numpy as np
from pymatgen.analysis.interfaces.zsl import fast_norm
from InterOptimus.CNID import triple_dot, calculate_cnid_in_supercell
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import os

def existfilehere(filename):
    return os.path.isfile(os.path.join(os.getcwd(), filename))

def get_termination_indices(slab, ftol= 0.25):
    """
    get the terminating atom indices of a slab.
    
    Args:
    (Structure): slab structure.
    
    Return:
    (arrays): terminating atom indices at the top and bottom.
    """
    frac_coords = slab.frac_coords
    n = len(frac_coords)
    dist_matrix = np.zeros((n, n))
    h = slab.lattice.c
    # Projection of c lattice vector in
    # direction of surface normal.
    for ii, jj in combinations(list(range(n)), 2):
        if ii != jj:
            cdist = frac_coords[ii][2] - frac_coords[jj][2]
            cdist = abs(cdist - np.round(cdist)) * h
            dist_matrix[ii, jj] = cdist
            dist_matrix[jj, ii] = cdist

    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, ftol, criterion="distance")
    clustered_sites: dict[int, list[Site]] = {c: [] for c in clusters}
    for idx, cluster in enumerate(clusters):
        clustered_sites[cluster].append(slab[idx])
    plane_heights = {np.mean(np.mod([s.frac_coords[2] for s in sites], 1)): c for c, sites in clustered_sites.items()}
    term_cluster_min = min(plane_heights.items(), key=lambda x: x[0])[1]
    term_cluster_max = max(plane_heights.items(), key=lambda x: x[0])[1]
    return np.where(clusters == term_cluster_min)[0], np.where(clusters == term_cluster_max)[0]

def get_termination_indices_shell(slab, shell = 1.5):
    """
    get the terminating atom indices of a slab.
    
    Args:
    (Structure): slab structure.
    shell(float): shell size to include termination atoms
    
    Return:
    (arrays): terminating atom indices at the top and bottom.
    """
    frac_coords_z = slab.cart_coords[:,2]
    low = min(frac_coords_z)
    high = max(frac_coords_z)
    return np.where(frac_coords_z < low + shell)[0], np.where(frac_coords_z > high - shell)[0]
    
def get_it_core_indices(interface):
    """
    get the terminating atom indices of a interface.
    
    Args:
    interface (Interface).
    
    Returns:
    (arrays): film top & bottom indices; substrate top & bottom indices.
    """
    ids = np.array(interface.film_indices)
    slab = interface.film
    ids_film_min, ids_film_max = ids[get_termination_indices(slab)[0]], ids[get_termination_indices(slab)[1]]
    
    ids = np.array(interface.substrate_indices)
    slab = interface.substrate
    ids_substrate_min, ids_substrate_max = ids[get_termination_indices(slab)[0]], ids[get_termination_indices(slab)[1]]
    return ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max


def convert_value(value):
    if value.upper() == '.TRUE.' or value.upper() == 'TRUE':
        return True
    elif value.upper() == '.FALSE.' or value.upper() == 'FALSE':
        return False
    if '/' in value:
        return value
    if ',' in value:
        return np.array(value.split(','), dtype = int)
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        return value

def read_key_item(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            if '=' in line:
                tag, value = line.split('=', 1)
                tag = tag.strip()
                value = value.strip()
                data[tag] = convert_value(value)
                
    if 'THEORETICAL' not in data.keys():
        data['THEORETICAL'] = False
    if 'STABLE' not in data.keys():
        data['STABLE'] = True
    if 'NOELEM' not in data.keys():
        data['NOELEM'] = True
    if 'STCTMP' not in data.keys():
        data['STCTMP'] = True
    return data
    
def get_one_interface(cib, termination, slab_length, xyz, vacuum_over_film, c_periodic = False):
    x,y,z = xyz
    if c_periodic:
        gap = vacuum_over_film = z
    else:
        gap = z
        vacuum_over_film = vacuum_over_film
    interface_here = list(cib.get_interfaces(termination= termination, \
                                   substrate_thickness = slab_length, \
                                   film_thickness=slab_length, \
                                   vacuum_over_film=vacuum_over_film, \
                                   gap=gap, \
                                   in_layers=False))[0]
    CNID = calculate_cnid_in_supercell(interface_here)[0]
    CNID_translation = TranslateSitesTransformation(interface_here.film_indices, x*CNID[:,0] + y*CNID[:,1])
    return CNID_translation.apply_transformation(interface_here)

def get_rot_strain(film_matrix, sub_matrix) -> np.ndarray:
    """Find transformation matrix that will rotate and strain the film to the substrate while preserving the c-axis."""
    film_matrix = np.array(film_matrix)
    film_matrix = film_matrix.tolist()[:2]
    film_matrix.append(np.cross(film_matrix[0], film_matrix[1]))
    film_matrix[2] = film_matrix[2]/np.linalg.norm(film_matrix[2])
    # Generate 3D lattice vectors for substrate super lattice
    # Out of plane substrate super lattice has to be same length as
    # Film out of plane vector to ensure no extra deformation in that
    # direction
    sub_matrix = np.array(sub_matrix)
    sub_matrix = sub_matrix.tolist()[:2]
    temp_sub = np.cross(sub_matrix[0], sub_matrix[1]).astype(float)  # conversion to float necessary if using numba
    temp_sub *= fast_norm(np.array(film_matrix[2], dtype=float))  # conversion to float necessary if using numba
    sub_matrix.append(temp_sub)
    sub_matrix[2] = sub_matrix[2]/np.linalg.norm(sub_matrix[2])

    A = np.transpose(np.linalg.solve(film_matrix, sub_matrix))
    U, sigma, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    
    S = np.dot(R.T, A)
    return R, S

def get_non_strained_film(match, it):
    f_vs = match.film_sl_vectors
    s_vs = match.substrate_sl_vectors
    R_21, s_21 = get_rot_strain(f_vs, s_vs)
    R_1it, _ = get_rot_strain(s_vs, it.lattice.matrix[:2])
    trans_f = np.dot(R_1it, R_21)
    trans_b = np.dot(np.linalg.inv(R_21), np.linalg.inv(R_1it))
    trans = triple_dot(trans_f, np.linalg.inv(s_21), trans_b)
    DST = DeformStructureTransformation(trans)
    return trans_to_bottom(DST.apply_transformation(it.film))

def trans_to_bottom(stct):
    ids = np.arange(len(stct))
    min_fc = stct.frac_coords[:,2].min()
    TST = TranslateSitesTransformation(ids, [0,0,-min_fc])
    return TST.apply_transformation(stct)

def get_film_length(match, film, it):
    film_sg = SlabGenerator(
            film,
            match.film_miller,
            min_slab_size=1,
            min_vacuum_size=10,
            in_unit_planes=False,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )
    return film_sg._proj_height * it.film_layers

def add_sele_dyn(it):
    sub_bot_indices = get_it_core_indices(it)[2]
    mobility_mtx = np.repeat(np.array([[True, True, True]]), len(it), axis = 0)
    mobility_mtx[sub_bot_indices] = [False, False, False]
    it.add_site_property('selective_dynamics', mobility_mtx)
    return it

def add_sele_dyn_slab(slab):
    sub_bot_indices = get_termination_indices(slab)[0]
    mobility_mtx = np.repeat(np.array([[True, True, True]]), len(slab), axis = 0)
    mobility_mtx[sub_bot_indices] = [False, False, False]
    slab.add_site_property('selective_dynamics', mobility_mtx)
    return slab
