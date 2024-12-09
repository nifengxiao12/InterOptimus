from InterOptimus.matching import interface_searching, EquiMatchSorter
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from pymatgen.core.structure import Structure
from pymatgen.analysis.interfaces import SubstrateAnalyzer
from InterOptimus.equi_term import get_non_identical_slab_pairs
from InterOptimus.tool import apply_cnid_rbt, trans_to_bottom, sort_list, get_it_core_indices, get_min_nb_distance
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from skopt import gp_minimize
from skopt.space import Real
from tqdm.notebook import tqdm
from numpy import array, dot, column_stack, argsort, zeros, mod, mean, ceil, concatenate, random, repeat
from numpy.linalg import norm
from InterOptimus.CNID import calculate_cnid_in_supercell
from InterOptimus.VaspWorkFlow import ItFireworkPatcher
import os
import pandas as pd
from mlipdockers.core import MlipCalc
from fireworks import Workflow
import json

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
    
    def set_energy_calculator_docker(self, calc):
        """
        set energy calculator docker container
        
        Args:
        calc (str): mace, orb-models, sevenn, chgnet, grace-2l
        """
        self.mc = MlipCalc(image_name = calc)
    
    def close_energy_calculator(self):
        """
        close energy calculator docker container
        """
        self.mc.close()
    
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
        term_atom_ids = self.get_interface_atom_indices(interface_here)
        for i in term_atom_ids:
            if get_min_nb_distance(i, interface_here, self.discut) < self.discut:
                return 0
        self.opt_results[(self.match_id_now,self.term_id_now)]['sampled_interfaces'].append(interface_here)
        return self.mc.calculate(interface_here)
        
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
    
    def get_interface_energy_and_binding_energy(self, supcl_E, match_id, term_id, area):
        """
        calculate interface energy & interface binding energy
        
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
        film_single_E = self.mc.calculate(single_pair[0])
        film_double_E = self.mc.calculate(double_pair[0])
        substrate_single_E = self.mc.calculate(single_pair[1])
        substrate_double_E = self.mc.calculate(double_pair[1])
        film_adhe_E = - film_double_E + 2 * film_single_E
        substrate_adhe_E = - substrate_double_E + 2 * substrate_single_E
        return (supcl_E - (film_double_E + substrate_double_E) / 2) / area * 16.02176634, \
               (supcl_E - (film_single_E + substrate_single_E)) / area * 16.02176634, single_pair, double_pair
    
    def get_interface_atom_indices(self, interface):
        """
        get the indices of interface atoms
        
        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        
        Return:
        indices (array)
        """
        #interface atom indices
        ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max = get_it_core_indices(interface)
        if self.c_periodic:
            return concatenate((ids_film_min, ids_film_max, ids_substrate_min, ids_substrate_max))
        else:
            return concatenate((ids_film_min, ids_substrate_max))
    
    def optimize_specified_interface_by_mlip(self, match_id, term_id, n_calls = 50, z_range = (0.5, 3), calc = 'mace'):
        """
        apply bassian optimization for the xyz registration of a specified interface with the predicted
        interface energy by machine learning potential

        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        calc: MLIP calculator (str): mace, orb-models, sevenn, chgnet, grace-2l
        """
        #initialize opt info dict
        if not hasattr(self, 'opt_results'):
            self.opt_results = {}
        self.opt_results[(match_id,term_id)] = {}
        self.opt_results[(match_id,term_id)]['sampled_interfaces'] = []

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
        self.opt_results[(match_id,term_id)]['sampled_interfaces'] = \
        sort_list(self.opt_results[(match_id,term_id)]['sampled_interfaces'], ys)

        #rank energy
        ys = ys[argsort(ys)]

        #get cartesian xyzs
        interface = self.get_specified_interface(match_id, term_id)
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = column_stack((dot(interface.lattice.matrix.T, CNID),[0,0,0]))
        xs_cart = dot(CNID_cart, xs.T).T + column_stack((zeros(len(xs)), zeros(len(xs)), xs[:,2]))
        
        self.opt_results[(match_id,term_id)]['xyzs_ognl'] = xs
        self.opt_results[(match_id,term_id)]['xyzs_cart'] = xs_cart
        self.opt_results[(match_id,term_id)]['supcl_E'] = ys
    
    def global_minimization(self, n_calls = 50, z_range = (0.5, 3), calc = 'mace', discut = 0.8):
        """
        apply bassian optimization for the xyz registration of all the interfaces with the predicted
        interface energy by machine learning potential, getting ranked interface energies

        Args:
        n_calls (int): number of calls
        z_range (tuple): sampling range of z
        calc (str): MLIP calculator: mace, orb-models, sevenn, chgnet, grace-2l
        discut: (float): allowed minimum atomic distance for searching
        """
        self.discut = discut
        columns = [r'$h_s$',r'$k_s$',r'$l_s$',
                  r'$h_f$',r'$k_f$',r'$l_f$',
                   r'$A$ (' + '\u00C5' + '$^2$)', r'$\epsilon$', r'$E_{it}$ $(J/m^2)$', r'$E_{bd}$ $(J/m^2)$', r'$E_{sp}$',
                   r'$u_{f1}$',r'$v_{f1}$',r'$w_{f1}$',
                   r'$u_{f2}$',r'$v_{f2}$',r'$w_{f2}$',
                   r'$u_{s1}$',r'$v_{s1}$',r'$w_{s1}$',
                   r'$u_{s2}$',r'$v_{s2}$',r'$w_{s2}$', r'$T$', r'$i_m$', r'$i_t$']
        formated_data = []
        #set docker container
        self.set_energy_calculator_docker(calc)
        #scanning matches and terminations
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            for i in range(len(self.unique_matches)):
                with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                    for j in range(len(self.all_unique_terminations[i])):
                        #optimize
                        self.optimize_specified_interface_by_mlip(i, j, n_calls = n_calls, z_range = z_range, calc = calc)
                        
                        #formated data
                        m = self.unique_matches
                        idt = self.unique_matches_indices_data
                        
                        hkl_f, hkl_s = m[i].film_miller, m[i].substrate_miller
                        A, epsilon, E_sup = m[i].match_area, m[i].von_mises_strain, self.opt_results[(i,j)]['supcl_E'][0]
                        uvw_f1, uvw_f2 = idt[i]['film_conventional_vectors']
                        uvw_s1, uvw_s2 = idt[i]['substrate_conventional_vectors']
                        
                        ##calculate adhesive & interface energy
                        it_E, bd_E, single_pair, double_pair = self.get_interface_energy_and_binding_energy(E_sup, i, j, A)
                        
                        ##save single double slabs
                        self.opt_results[(i,j)]['single_slabs'] = {}
                        self.opt_results[(i,j)]['single_slabs']['film'] = single_pair[0]
                        self.opt_results[(i,j)]['single_slabs']['substrate'] = single_pair[1]
                        
                        self.opt_results[(i,j)]['double_slabs'] = {}
                        self.opt_results[(i,j)]['double_slabs']['film'] = double_pair[0]
                        self.opt_results[(i,j)]['double_slabs']['substrate'] = double_pair[1]
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
        
        #close docker container
        self.close_energy_calculator()

    def random_sampling_specified_interface(self, match_id, term_id, n_taget, n_max, sampling_min_displace, discut):
        """
        perform random sampling of rigid body translation for a specified interface
        
        Args:
        match_id (int): unique match id
        term_id (int): unique term id
        n_taget (int): target number of sampling
        n_max (int): max number of trials
        sampling_min_displace (float): sampled rigid body translation position are not allowed to be closer than this (angstrom)
        discut (float): the atoms are not allowed to be closer than this (angstrom)
        
        Return:
        sampled_interfaces (list): list of sampled interfaces (json)
        xyzs (list): list of sampled xyz parameters
        rbt_carts: list of sampled RBT positions in cartesian coordinates
        """
        #get initial interface
        interface = self.get_specified_interface(match_id, term_id)
        #calculate cnid catesian
        CNID = calculate_cnid_in_supercell(interface)[0]
        CNID_cart = dot(interface.lattice.matrix.T, CNID)
        #sampling
        num_of_sampled = 1
        n_trials = 0
        rbt_carts = [[0,0,2]]
        xyzs = [[0,0,2]]
        ##interface atom indices
        sampled_interfaces = []
        sampled_interfaces.append(self.get_specified_interface(match_id, term_id, [0,0,2]).to_json())
        while num_of_sampled < n_taget and num_of_sampled < n_max:
            #sampling from (0,0,0) to (1,1,1)
            x,y,z = [random.random() for i in range(3)]
            #z is cartesian
            z = z * 3
            #calculate cartesian RBT
            cart_here = x*CNID_cart[:,0] + y*CNID_cart[:,1] + [0,0,z]
            #calculate distances between this RBT position and already sampled RBT positions
            distwithbefore = norm(repeat([cart_here], num_of_sampled, axis = 0) - rbt_carts, axis = 1)
            #RBT position distance not too close
            if min(distwithbefore) > sampling_min_displace:
                #min atomic distance not too close
                interface_here = self.get_specified_interface(match_id, term_id, [x, y, z])
                existing_too_close_sites = False
                ##interface atomic indices
                it_atom_ids = self.get_interface_atom_indices(interface_here)
                for i in it_atom_ids:
                    if get_min_nb_distance(i, interface_here, discut) < discut:
                        existing_too_close_sites = True
                        break
                if not existing_too_close_sites:
                    #interface_here.to_file(f'op_its/{num_of_sampled}_POSCAR')
                    sampled_interfaces.append(interface_here.to_json())
                    rbt_carts.append(list(cart_here))
                    xyzs.append([x,y,z])
                    num_of_sampled += 1
            n_trials += 1
        
        return sampled_interfaces, xyzs, rbt_carts
        
    def global_random_sampling(self, n_taget, n_max, sampling_min_displace, discut, to_fireworks = False, **kwargs):
        """
        perform random sampling of rigid body translation for all the interface
        
        Args:
        n_taget (int): target number of sampling
        n_max (int): max number of trials
        sampling_min_displace (float): sampled rigid body translation position are not allowed to be closer than this (angstrom)
        discut (float): the atoms are not allowed to be closer than this (angstrom)
        to_fireworks (bool): whether to generate firework workflow dict
        
        kwargs:
        project_name (str): project name to be stored in mongodb database
        db_file (str): path to atomate mongodb config file
        vasp_cmd (str): command to run vasp
        work_dir (str): working directory
        update_incar_settings, update_potcar_settings, update_kpoints_settings (dict): user incar, potcar, kpoints settings
        update_potcar_functional (str): which set of functional to use

        Return:
        (Workflow)
        """
        self.global_random_sample_dict = {}
        with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
            for i in range(len(self.unique_matches)):
                with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                    for j in range(len(self.all_unique_terminations[i])):
                        key = f'{i}_{j}'
                        self.global_random_sample_dict[key] = {}
                        
                        self.global_random_sample_dict[key]['sampled_interfaces'], \
                        self.global_random_sample_dict[key]['xyzs'], \
                        self.global_random_sample_dict[key]['rbt_carts'] \
                        = self.random_sampling_specified_interface(i, j, n_taget, n_max, \
                                                                    sampling_min_displace, discut)
                        
                        term_pbar.update(1)
                match_pbar.update(1)
                
        if to_fireworks:
            with open('global_random_sampling.json','w') as f:
                json.dump(self.global_random_sample_dict, f)
            for st in ['user_incar_settings', 'user_potcar_settings', 'user_kpoints_settings', 'user_potcar_functional']:
                if st not in kwargs.keys():
                    kwargs[st] = None
            it_firework_patcher = ItFireworkPatcher(kwargs['project_name'], kwargs['db_file'], kwargs['vasp_cmd'],
                                                     user_incar_settings = kwargs['user_incar_settings'],
                                                     user_potcar_settings = kwargs['user_potcar_settings'],
                                                     user_kpoints_settings = kwargs['user_kpoints_settings'],
                                                     user_potcar_functional = kwargs['user_potcar_functional'])
            wf = []
            with tqdm(total = len(self.unique_matches), desc = "matches") as match_pbar:
                for i in range(len(self.unique_matches)):
                    with tqdm(total = len(self.all_unique_terminations[i]), desc = "unique terminations") as term_pbar:
                        for j in range(len(self.all_unique_terminations[i])):
                            #slab fws
                            single_pairs, double_pairs = self.get_decomposition_slabs(i, j)
                            fws_fmsg = it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static', single_pairs[0],
                                                                                                {'i':i, 'j':j, 'tp':'fmsg'},
                                                                                                os.path.join(kwargs['work_dir'], f'fmsg_{i}_{j}'))
                                                                                                
                            fws_fmdb = it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static', double_pairs[0],
                                                                                                {'i':i, 'j':j, 'tp':'fmdb'},
                                                                                                os.path.join(kwargs['work_dir'], f'fmdb_{i}_{j}'))
                                                                                                
                            fws_stsg = it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static', single_pairs[1],
                                                                                                {'i':i, 'j':j, 'tp':'stsg'},
                                                                                                os.path.join(kwargs['work_dir'], f'stsg_{i}_{j}'))
                                                                                                
                            fws_stdb = it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static', double_pairs[1],
                                                                                                {'i':i, 'j':j, 'tp':'stdb'},
                                                                                                os.path.join(kwargs['work_dir'], f'stdb_{i}_{j}'))
                            wf += fws_fmsg + fws_fmdb + fws_stsg + fws_stdb
                            
                            #interface fws
                            its = self.global_random_sample_dict[f'{i}_{j}']['sampled_interfaces']
                            for k in range(len(its)):
                                fws = it_firework_patcher.non_dipole_mod_fol_by_diple_mod('interface static', Structure.from_dict(json.loads(its[k])),
                                                                                                {'i':i, 'j':j, 'k':k, 'tp':'it'},
                                                                                                os.path.join(kwargs['work_dir'], f'it_{i}_{j}_{k}'))
                                wf += fws
                                
                            term_pbar.update(1)
                    match_pbar.update(1)
        return Workflow(wf)
