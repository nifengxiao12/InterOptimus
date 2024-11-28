from InterOptimus.matching import interface_searching, EquiMatchSorter
from InterOptimus.optimize import WorkPatcher, InputDataGenerator, interface_score_ranker
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces import SubstrateAnalyzer
from InterOptimus.equi_term import get_non_identical_slab_pairs
from InterOptimus.tool import apply_cnid_rbt, trans_to_bottom, sort_list
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from chgnet.model.model import CHGNet
from skopt import gp_minimize
from skopt.space import Real
from tqdm.notebook import tqdm
from numpy import array, dot, column_stack, argsort, zeros, mod, mean, ceil
from InterOptimus.CNID import calculate_cnid_in_supercell
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.sets import MPStaticSet, MPRelaxSet
import os
import pandas as pd

def get_env_variable(var_name):
    try:
        value = os.environ[var_name]
        return value
    except KeyError:
        raise EnvironmentError(f"Environment variable '{var_name}' is not set.")

def update_setting_dict(old_dict, new_dict):
    """
    update old dict by new dict

    Args:
    old_dict (dict)
    new_dict (dict)
    """
    if new_dict != None:
        for i in new_dict.keys():
            old_dict[i] = new_dict[i]
    return old_dict

def get_potcar(structure, functional):
    """
    get the pymatgen Potcar instance for a structure

    Args:
    structure (Structure)
    functional (str): functional set to use
    
    Return:
    (Potcar)
    """
    return Potcar([get_potcar_dict()[i.symbol] for i in structure.elements], functional)

def get_potcar_dict():
    """
    default potcar label for each element (used recommended ones by the vasp mannual for the PBE.54 functionals)

    Return:
    (dict): {element: potcar label}
    """
    return {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 'Ar': 'Ar', 'As': 'As', 'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be_sv', 'Bi': 'Bi', 'Br': 'Br', 'C': 'C', 'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 'Cl': 'Cl', 'Co': 'Co', 'Cr': 'Cr_pv', 'Cs': 'Cs_sv', 'Cu': 'Cu_pv', 'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu', 'F': 'F', 'Fe': 'Fe_pv', 'Ga': 'Ga_d', 'Gd': 'Gd', 'Ge': 'Ge_d', 'H': 'H', 'He': 'He', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv', 'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 'Mg': 'Mg_pv', 'Mn': 'Mn_pv', 'Mo': 'Mo_pv', 'N': 'N', 'Na': 'Na_pv', 'Nb': 'Nb_pv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Ni': 'Ni', 'Np': 'Np', 'O': 'O', 'Os': 'Os_pv', 'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd', 'Pm': 'Pm_3', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'Rb': 'Rb_sv', 'Re': 'Re_pv', 'Rh': 'Rh_pv', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 'Ti': 'Ti_pv', 'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 'V': 'V_pv', 'W': 'W_pv', 'Xe': 'Xe', 'Y': 'Y_sv', 'Yb': 'Yb_2', 'Zn': 'Zn', 'Zr': 'Zr_sv'}

def get_default_incar_settings(name, **kwargs):
    """
    get default incar settings

    Args:
    name (str): what to calculate
    """
    if name == 'standard relax':
        return {
        "ALGO": "Normal",
        "LDAU": False,
        "EDIFF": 1e-5,
        "ISIF": 3,
        "NELM": 1000,
        "NSW": 400,
        "PREC": None,
        "EDIFFG": -0.02,
        "ISPIN": 2,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        }
    elif name == 'standard static':
        return {
        "LDAU": False,
        "EDIFF": 1e-5,
        "NELM": 1000,
        "PREC": None,
        "ISPIN": 2,
        "ALGO": "Normal",
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LWAVE": True,
        "LREAL": "Auto"}
    elif name == 'interface static':
        incar_settings = get_default_incar_settings('standard static')
        if kwargs['LDIPOL']:
            incar_settings['LDIPOL'] = True
            incar_settings['IDIPOL'] = 3
        return incar_settings
    elif name == 'interface relax':
        incar_settings = get_default_incar_settings('standard relax')
        if kwargs['LDIPOL']:
            incar_settings['LDIPOL'] = True
            incar_settings['IDIPOL'] = 3
        if kwargs['c_periodic']:
            incar_settings['IOPTCELL'] = "0 0 0 0 0 0 0 0 1"
        else:
            incar_settings['ISIF'] = 2
        return incar_settings
    else:
        raise ValueError("default inter settings only support 'standard relax', 'standard static', 'interface static' and 'interface relax'")
        
def get_vasp_input_settings(name, structure, ENCUT_scale = 1.3, update_incar_settings = None, update_potcar_functional = None,
                            update_potcar_settings = None, update_kpoints_settings = None, **kwargs):
    
    """
    get user vasp input settings
    
    Args:
    name (str): one of 'standard relax', 'standard static', 'interface static', 'interface relax'
    structure (Structure): structure for calculation
    ENCUT_scale (float): scaling factor of the maximum en_cut in potcars
    update_incar_settings, update_potcar_settings, update_kpoints_settings (dict): user incar, potcar, kpoints settings
    update_potcar_functional (str): which set of functional to use
    
    Return:
    (VaspInputSet)
    """
    default_incar_settings = get_default_incar_settings(name, **kwargs)
    default_potcar_settings = get_potcar_dict()
    default_kpoints_settings = {'reciprocal_density': 200}
    user_incar_settings = update_setting_dict(default_incar_settings, update_incar_settings)
    user_potcar_settings = update_setting_dict(default_potcar_settings, update_potcar_settings)
    user_kpoints_settings = update_setting_dict(default_kpoints_settings, update_kpoints_settings)

    if update_potcar_functional != None:
        user_potcar_functional = 'PBE_54'
    else:
        user_potcar_functional = update_potcar_functional

    potcar = get_potcar(structure, user_potcar_functional)
    max_encut = max(p.keywords['ENMAX'] for p in potcar)
    custom_encut = max_encut * ENCUT_scale
    user_incar_settings['ENCUT'] = custom_encut
    if name == 'standard relax' or name == 'interface relax':
        return MPRelaxSet(structure, user_incar_settings = user_incar_settings, \
                                       user_potcar_functional=user_potcar_functional, \
                                       user_potcar_settings = user_potcar_settings, \
                                       user_kpoints_settings = user_kpoints_settings)
    else:
        return MPStaticSet(structure, user_incar_settings = user_incar_settings, \
                                       user_potcar_functional=user_potcar_functional, \
                                       user_potcar_settings = user_potcar_settings, \
                                       user_kpoints_settings = user_kpoints_settings)

def registration_minimizer(interfaceworker, n_calls, z_range):
    """
    baysian optimization for xyz registration
    
    Args:
    n_calls (int): num of optimization
    z_range (float): range of z sampling
    
    Return:
    optimization result
    """
    def trial_with_progress(func, n_calls, *args, **kwargs):
        with tqdm(total = n_calls, desc = "registration optimizing") as rgst_pbar:  # Initialize tqdm with total number of iterations
            def wrapped_func(*args, **kwargs):
                result = func(*args, **kwargs)
                rgst_pbar.update(1)  # Update progress bar by 1 after each function call
                return result
            return gp_minimize(wrapped_func, search_space, n_calls=n_calls, *args, **kwargs)
    search_space = [
        Real(0, 1, name='x'),
        Real(0, 1, name='y'),
        Real(z_range[0], z_range[1], name = 'z')
    ]
    # Run the optimization with progress bar
    result = trial_with_progress(interfaceworker.sample_xyz_energy, n_calls=n_calls, random_state=42)
    return result

class InterfaceWorker:
    """
    core class for the interface jobs
    """
    def __init__(self, film_conv, substrate_conv):
        """
        Args:
        film_conv (Structure): film conventional cell
        substrate_conv (Structure): substrate conventional cell
        """
        self.film_conv = film_conv
        self.substrate_conv = substrate_conv
        self.film = film_conv.get_primitive_structure()
        self.substrate = substrate_conv.get_primitive_structure()
        
    def lattice_matching(self, max_area = 47, max_length_tol = 0.03, max_angle_tol = 0.01,
                         film_max_miller = 3, substrate_max_miller = 3, film_millers = None, substrate_millers = None):
        """
        lattice matching by Zur and McGill

        Args:
        max_area (float), max_length_tol (float), max_angle_tol (float): searching tolerance parameters
        film_max_miller (int), substrate_max_miller (int): maximum miller index
        film_millers (None|array), substrate_millers (None|array): specified searching miller indices (optional)
        """
        sub_analyzer = SubstrateAnalyzer(max_area = max_area, max_length_tol = max_length_tol, max_angle_tol = max_angle_tol,
                                         film_max_miller = film_max_miller, substrate_max_miller = substrate_max_miller)
        self.unique_matches, \
        self.equivalent_matches, \
        self.unique_matches_indices_data,\
        self.equivalent_matches_indices_data,\
        self.areas = interface_searching(self.substrate_conv, self.film_conv, sub_analyzer, film_millers, substrate_millers)
        self.ems = EquiMatchSorter(self.film_conv, self.substrate_conv, self.equivalent_matches_indices_data, self.unique_matches)

    def parse_interface_structure_params(self, termination_ftol = 0.01, c_periodic = False, \
                                        vacuum_over_film = 10, film_thickness = 10, substrate_thickness = 10, shift_to_bottom = True):
        """
        parse necessary structure parameters for interface generation in the next steps

        Args:

        termination_ftol (float): tolerance of the c-fractional coordinates for termination atom clustering
        c_periodic (bool): whether to make double interface supercell
        vacuum_over_film (float): vacuum thickness over film
        film_thickness (float): film slab thickness
        substrate_thickness (float): substrate slab thickness
        shift_to_bottom (bool): whether to shift the supercell to the bottom
        """
        self.termination_ftol, self.c_periodic, self.vacuum_over_film, self.film_thickness, self.substrate_thickness, self.shift_to_bottom = \
        termination_ftol, c_periodic, vacuum_over_film, film_thickness, substrate_thickness, shift_to_bottom
        self.get_all_unique_terminations()

    def get_specified_match_cib(self, id):
        """
        get the CoherentInterfaceBuilder instance for a specified unique match

        Args:
        id (int): unique match index
        """
        cib = CoherentInterfaceBuilder(film_structure=self.film,
                               substrate_structure=self.substrate,
                               film_miller=self.unique_matches[id].film_miller,
                               substrate_miller=self.unique_matches[id].substrate_miller,
                               zslgen=SubstrateAnalyzer(max_area=200), termination_ftol=self.termination_ftol, label_index=True,\
                               filter_out_sym_slabs=False)
        cib.zsl_matches = [self.unique_matches[id]]
        return cib
    
    def get_unique_terminations(self, id):
        """
        get non-identical terminations for a specified unique match id

        Args:
        id (int): unique match index
        """
        unique_term_ids = get_non_identical_slab_pairs(self.film, self.substrate, self.unique_matches[id], \
                                                       ftol = self.termination_ftol, c_periodic = self.c_periodic)[0]
        cib = self.get_specified_match_cib(id)
        return [cib.terminations[i] for i in unique_term_ids]
    
    def get_all_unique_terminations(self):
        """
        get unique terminations for all the unique matches
        """
        all_unique_terminations = []
        for i in range(len(self.unique_matches)):
            all_unique_terminations.append(self.get_unique_terminations(i))
        self.all_unique_terminations = all_unique_terminations
    
    def get_specified_interface(self, match_id, term_id, xyz = [0,0,2]):
        """
        get a specified interface by unique match index, unique termination index, and xyz registration

        Args:
        match_id (int): unique match index
        term_id (int): unique termination index
        xyz (array): xyz registration
        
        Return:
        (Interface)
        """
        x, y, z = xyz
        if self.c_periodic:
            gap = vacuum_over_film = z
        else:
            gap = z
            vacuum_over_film = self.vacuum_over_film
        cib = self.get_specified_match_cib(match_id)
        interface_here = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = self.substrate_thickness, film_thickness = self.film_thickness, \
                                       vacuum_over_film = vacuum_over_film, gap = gap, in_layers = False))[0]
        interface_here = apply_cnid_rbt(interface_here, x, y, 0)
        if self.shift_to_bottom:
            interface_here = trans_to_bottom(interface_here)
        return interface_here

    def load_chgnet(self):
        """
        load chgnet module
        """
        self.chgnet = CHGNet.load()
        
    def sample_xyz_energy(self, params):
        """
        sample the predicted energy for a specified xyz registration of a initial interface
        
        Args:
        xyz: sampled xyz

        Return
        energy (float): predicted energy by chgnet
        """
        x,y,z = params
        xyz = [x,y,z]
        if self.c_periodic:
            interface_here = self.get_specified_interface(self.match_id_now, self.term_id_now, xyz = xyz)
        else:
            initial_interface = self.get_specified_interface(self.match_id_now, self.term_id_now, [0,0,2])
            xyz[2] = (xyz[2] - 2)/initial_interface.lattice.c
            interface_here = apply_cnid_rbt(initial_interface, xyz[0],xyz[1],xyz[2])
        self.chg_opt_results[(self.match_id_now,self.term_id_now)]['sampled_interfaces'].append(interface_here)
        return self.chgnet.predict_structure(interface_here)['e'] * len(interface_here)
            
    def optimize_specified_interface_by_chgnet(self, match_id, term_id, n_calls = 50, z_range = (0.5, 3)):
        """
        apply bassian optimization for the xyz registration of a specified interface

        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        """
        #initialize opt info dict
        if not hasattr(self, 'chg_opt_results'):
            self.chg_opt_results = {}
        self.chg_opt_results[(match_id,term_id)] = {}
        self.chg_opt_results[(match_id,term_id)]['sampled_interfaces'] = []

        #set match&term id
        self.match_id_now = match_id
        self.term_id_now = term_id

        #optimize
        result = registration_minimizer(self, n_calls, z_range)
        xs = array(result.x_iters)
        ys = result.func_vals

        #rank xs by energy
        xs = xs[argsort(ys)]
        
        #list need to be ranked by special function
        self.chg_opt_results[(match_id,term_id)]['sampled_interfaces'] = \
        sort_list(self.chg_opt_results[(match_id,term_id)]['sampled_interfaces'], ys)

        #rank energy
        ys = ys[argsort(ys)]

        #get cartesian xyzs
        interface = self.get_specified_interface(match_id, term_id)
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = column_stack((dot(interface.lattice.matrix.T, CNID),[0,0,0]))
        xs_cart = dot(CNID_cart, xs.T).T + column_stack((zeros(len(xs)), zeros(len(xs)), xs[:,2]))
        
        self.chg_opt_results[(match_id,term_id)]['xyzs_ognl'] = xs
        self.chg_opt_results[(match_id,term_id)]['xyzs_cart'] = xs_cart
        self.chg_opt_results[(match_id,term_id)]['supcl_E'] = ys
        
    def get_film_substrate_layer_thickness(self, match_id, term_id):
        """
        get single layer thickness
        """
        cib = self.get_specified_match_cib(match_id)
        interface_film_1 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = 2, film_thickness = 2, \
                                       vacuum_over_film = 0, gap = 0, in_layers = True))[0]
        interface_film_2 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = 2, film_thickness = 3, \
                                       vacuum_over_film = 0, gap = 0, in_layers = True))[0]

        interface_substrate_1 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = 2, film_thickness = 2, \
                                       vacuum_over_film = 0, gap = 0, in_layers = True))[0]
        interface_substrate_2 = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = 3, film_thickness = 2, \
                                       vacuum_over_film = 0, gap = 0, in_layers = True))[0]
        return interface_film_2.lattice.c - interface_film_1.lattice.c, interface_substrate_2.lattice.c - interface_substrate_1.lattice.c
    
    def get_decomposition_slabs(self, match_id, term_id):
        """
        get decomposed film & substrate slabs to calculate binding energy

        Args:
        match_id (int): unique match index
        term_id (int): unique termination index

        Return:
        (single_film, single_substrate), (double_film, double_substrate) (tuple): single and double (film, substrate) pairs
        """
        
        cib = self.get_specified_match_cib(match_id)

        film_dx, substrate_dx = self.get_film_substrate_layer_thickness(match_id, term_id)
        film_layers = int(ceil(self.film_thickness/film_dx))
        substrate_layers = int(ceil(self.substrate_thickness/substrate_dx))

        film_thickness_double = film_layers * 2 * film_dx - 0.1
        substrate_thickness_double = substrate_layers * 2 * substrate_dx - 0.1

        interface_single = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = self.substrate_thickness, film_thickness = self.film_thickness, \
                                       vacuum_over_film = self.vacuum_over_film, gap = 2, in_layers = False))[0]
        
        interface_double = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = substrate_thickness_double, film_thickness = film_thickness_double, \
                                       vacuum_over_film = self.vacuum_over_film, gap = 2, in_layers = False))[0]
        
        dx = interface_double.lattice.c - interface_single.lattice.c
        interface_double = list(cib.get_interfaces(termination = self.all_unique_terminations[match_id][term_id], \
                                       substrate_thickness = substrate_thickness_double, film_thickness = film_thickness_double, \
                                       vacuum_over_film = self.vacuum_over_film - dx/2, gap = 2, in_layers = False))[0]
        
        return (trans_to_bottom(interface_single.film), trans_to_bottom(interface_single.substrate)), \
                (trans_to_bottom(interface_double.film), trans_to_bottom(interface_double.substrate))
    
    def get_interface_binding_energy(self, supcl_E, match_id, term_id, area):
        """
        calculate interface energy
        
        Args:
        supcl_E: interface supercell energy
        match_id (int): unique match index
        term_id (int): unique termination index
        
        Return:
        (float): interface energy
        (float): binding energy
        single_pair, double_pair (tuple): slab pairs
        """
        single_pair, double_pair = self.get_decomposition_slabs(match_id, term_id)
        film_single_E = self.chgnet.predict_structure(single_pair[0])['e'] * len(single_pair[0])
        film_double_E = self.chgnet.predict_structure(double_pair[0])['e'] * len(double_pair[0])
        substrate_single_E = self.chgnet.predict_structure(single_pair[1])['e'] * len(single_pair[1])
        substrate_double_E = self.chgnet.predict_structure(double_pair[1])['e'] * len(double_pair[1])
        film_adhe_E = - film_double_E + 2 * film_single_E
        substrate_adhe_E = - substrate_double_E + 2 * substrate_single_E
        return (supcl_E - (film_double_E + substrate_double_E) / 2) / area * 16.02176634, \
               (supcl_E - (film_single_E + substrate_single_E)) / area * 16.02176634, single_pair, double_pair
    
    def global_minimization(self, n_calls = 50, z_range = (0.5, 3)):
        columns = [r'$h_s$',r'$k_s$',r'$l_s$',
                  r'$h_f$',r'$k_f$',r'$l_f$',
                   r'$A$ (' + '\u00C5' + '$^2$)', r'$\epsilon$', r'$E_{it}$ $(J/m^2)$', r'$E_{bd}$ $(J/m^2)$', r'$E_{sp}$',
                   r'$u_{f1}$',r'$v_{f1}$',r'$w_{f1}$',
                   r'$u_{f2}$',r'$v_{f2}$',r'$w_{f2}$',
                   r'$u_{s1}$',r'$v_{s1}$',r'$w_{s1}$',
                   r'$u_{s2}$',r'$v_{s2}$',r'$w_{s2}$', r'$T$', r'$i_m$', r'$i_t$']
        formated_data = []
        self.load_chgnet()
        #scanning matches and terminations
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            for i in range(len(self.unique_matches)):
                with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                    for j in range(len(self.all_unique_terminations[i])):
                        #optimize
                        self.optimize_specified_interface_by_chgnet(i, j, n_calls = n_calls, z_range = z_range)
                        
                        #formated data
                        m = self.unique_matches
                        idt = self.unique_matches_indices_data
                        
                        hkl_f, hkl_s = m[i].film_miller, m[i].substrate_miller
                        A, epsilon, E_sup = m[i].match_area, m[i].von_mises_strain, self.chg_opt_results[(i,j)]['supcl_E'][0]
                        uvw_f1, uvw_f2 = idt[i]['film_conventional_vectors']
                        uvw_s1, uvw_s2 = idt[i]['substrate_conventional_vectors']
                        
                        ##calculate adhesive & interface energy
                        it_E, bd_E, single_pair, double_pair = self.get_interface_binding_energy(E_sup, i, j, A)
                        
                        ##save single double slabs
                        self.chg_opt_results[(i,j)]['single_slabs'] = {}
                        self.chg_opt_results[(i,j)]['single_slabs']['film'] = single_pair[0]
                        self.chg_opt_results[(i,j)]['single_slabs']['substrate'] = single_pair[1]
                        
                        self.chg_opt_results[(i,j)]['double_slabs'] = {}
                        self.chg_opt_results[(i,j)]['double_slabs']['film'] = double_pair[0]
                        self.chg_opt_results[(i,j)]['double_slabs']['substrate'] = double_pair[1]
                        formated_data.append(
                                    [hkl_f[0], hkl_f[1], hkl_f[2],\
                                    hkl_s[0], hkl_s[1], hkl_s[2], \
                                    A, epsilon, it_E, bd_E, E_sup, \
                                    uvw_f1[0], uvw_f1[1], uvw_f1[2], \
                                    uvw_f2[0], uvw_f2[1], uvw_f2[2], \
                                    uvw_s1[0], uvw_s1[1], uvw_s1[2], \
                                    uvw_s2[0], uvw_s2[1], uvw_s2[2], self.all_unique_terminations[i][j], i, j])
                        term_pbar.update(1)
                match_pbar.update(1)
                
        self.global_optimized_data = pd.DataFrame(formated_data, columns = columns)
        self.global_optimized_data = self.global_optimized_data.sort_values(by = r'$E_{it}$ $(J/m^2)$')
