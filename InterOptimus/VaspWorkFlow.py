from fireworks import FireTaskBase, FWAction, explicit_serialize, Firework, Workflow
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet, ModifyIncar
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.analysis.interfaces.coherent_interfaces import get_rot_3d_for_2d
from scipy.linalg import polar
from CNID import calculate_cnid_in_supercell
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from numpy import arange, ceil, savetxt, dot, meshgrid, array
from numpy.linalg import norm
import pickle
import shutil
import os
from tool import get_one_interface

def get_potcar_dict():
    return {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 'Ar': 'Ar', 'As': 'As', 'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be_sv', 'Bi': 'Bi', 'Br': 'Br', 'C': 'C', 'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 'Cl': 'Cl', 'Co': 'Co', 'Cr': 'Cr_pv', 'Cs': 'Cs_sv', 'Cu': 'Cu_pv', 'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu', 'F': 'F', 'Fe': 'Fe_pv', 'Ga': 'Ga_d', 'Gd': 'Gd', 'Ge': 'Ge_d', 'H': 'H', 'He': 'He', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv', 'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 'Mg': 'Mg_pv', 'Mn': 'Mn', 'Mo': 'Mo_pv', 'N': 'N', 'Na': 'Na_pv', 'Nb': 'Nb_pv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Ni': 'Ni', 'Np': 'Np', 'O': 'O', 'Os': 'Os_pv', 'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd', 'Pm': 'Pm_3', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'Rb': 'Rb_sv', 'Re': 'Re_pv', 'Rh': 'Rh_pv', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 'Ti': 'Ti_pv', 'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 'V': 'V_pv', 'W': 'W_pv', 'Xe': 'Xe', 'Y': 'Y_sv', 'Yb': 'Yb_2', 'Zn': 'Zn', 'Zr': 'Zr_sv'}

def get_potcar(structure):
    return Potcar([get_potcar_dict()[i.symbol] for i in structure.elements])

def CstRelaxSet(structure, ENCUT_scale = 1, NCORE = 12, Kdense = 1000):
    potcar = get_potcar(structure)
    max_encut = max(p.keywords['ENMAX'] for p in potcar)
    custom_encut = max_encut * ENCUT_scale
    user_incar_settings = {
        "ENCUT": custom_encut,
        "ALGO": "Normal",
        "LDAU": False,
        "EDIFF": 1e-5,
        "ISIF": 3,
        "NELM": 1000,
        "NSW": 10000,
        "PREC": None,
        "EDIFFG": -0.01,
        "NCORE": NCORE,
        "ISPIN": 2,
        "ISMEAR": 0,
        "SIGMA": 0.05,
    }
    return MPRelaxSet(structure, user_incar_settings = user_incar_settings, \
    user_potcar_functional='PBE_54', user_potcar_settings = get_potcar_dict(), user_kpoints_settings = {'reciprocal_density': Kdense})

def ITRelaxSet(structure, ENCUT_scale = 1, NCORE = 12, LDIPOL = True, c_periodic = False, EDIFF = 1e-4, Kdense = 500):
    potcar = get_potcar(structure)
    max_encut = max(p.keywords['ENMAX'] for p in potcar)
    custom_encut = max_encut * ENCUT_scale
    if LDIPOL:
        IDIPOL = 3
    else:
        IDIPOL = None
    if c_periodic:
        IOPTCELL = "0 0 0 0 0 0 0 0 1"
        ISIF = 3
    else:
        IOPTCELL = None
        ISIF = 2
    user_incar_settings = {
        "ENCUT": custom_encut,
        "ALGO": "Normal",
        "LDAU": False,
        "EDIFF": EDIFF,
        "ISIF": ISIF,
        "NELM": 1000,
        "NSW": 10000,
        "PREC": None,
        "EDIFFG": -0.05,
        "LDIPOL": LDIPOL,
        "IDIPOL": IDIPOL,
        "IOPTCELL": "0 0 0 0 0 0 0 0 1",
        "NCORE": NCORE,
        "ISPIN": 2,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LREAL": "Auto"
    }
    return MPRelaxSet(structure, user_incar_settings = user_incar_settings, \
    user_potcar_functional='PBE_54', user_potcar_settings = get_potcar_dict(), user_kpoints_settings = {'reciprocal_density': Kdense})

def ITStaticSet(structure, ENCUT_scale = 1, NCORE = 12, LDIPOL = True, EDIFF = 1e-5, Kdense = 500):
    potcar = get_potcar(structure)
    max_encut = max(p.keywords['ENMAX'] for p in potcar)
    custom_encut = max_encut * ENCUT_scale
    if LDIPOL:
        IDIPOL = 3
    else:
        IDIPOL = None
    user_incar_settings = {
        "ENCUT": custom_encut,
        "LDAU": False,
        "EDIFF": EDIFF,
        "NELM": 1000,
        "PREC": None,
        "LDIPOL": LDIPOL,
        "IDIPOL": IDIPOL,
        "NCORE": NCORE,
        "ISPIN": 2,
        "ALGO": "Normal",
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LWAVE": True,
        "LREAL": "Auto"
    }
    return MPStaticSet(structure, user_incar_settings = user_incar_settings, \
    user_potcar_functional='PBE_54', user_potcar_settings = get_potcar_dict(), user_kpoints_settings = {'reciprocal_density': Kdense})
    
def get_initial_film(interface):
    """
    get the non-deformed film
    """
    sub_vs0 = interface.interface_properties['substrate_sl_vectors']
    film_vs0 = interface.interface_properties['film_sl_vectors']
    sub_vs1 = interface.lattice.matrix[:2]
    R1 = get_rot_3d_for_2d(sub_vs0, sub_vs1)
    original_trans = cib.zsl_matches[0].match_transformation
    R0, T = polar(original_trans)
    strain_inv = dot(dot(R, inv(T)), inv(R))
    new_lattice = np.dot(strain_inv, interface.lattice.matrix.T).T
    return Structure(new_lattice, interface.film.species, interface.film.frac_coords)

def get_film_c_length(interface, in_unit_planes):
    """
    get the normal length of film slab
    """
    film_sg = SlabGenerator(
            cib.film_structure,
            interface.interface_properties['film_miller'],
            min_slab_size=interface.interface_properties['min_slab_size'],
            min_vacuum_size=1e-16,
            in_unit_planes=in_unit_planes,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )
    return norm(film_sg.get_slab(shift = 0).get_orthogonal_c_slab().lattice.matrix[2]) - 1e-16

def SlabEnergyWorkflows(self, interface, unique_match_idx, project_name, user_spec_sets = None):
    """
    work flow to calculate slab energy
    """
    spec = {"_nodes": 1, \
    "_ntasks_per_node": 64, \
    "_queue": "standard", \
    "_vasp_version":"6.4.3-optcell",\
    "_category":"vasp"}
    if user_spec_sets != None:
        for i in user_spec_sets.keys():
            spec[i] = user_spec_sets[i]
    film_slab = interface.film
    substrate_slab = interface.substrate
    film0_slab = get_initial_film(interface)
    
    film_slab_op_fw = Firework([
    WriteVaspFromIOSet(vasp_input_set=ITRelaxSet, structure=interface.film),
    RunVaspCustodian(),
    VaspToDb()], name=f"Optimization", spec = spec)
    
    substrate_slab_op_fw = Firework([
    WriteVaspFromIOSet(vasp_input_set=ITRelaxSet, structure=interface.substrate),
    RunVaspCustodian(),
    VaspToDb()], name=f"Optimization", spec = spec)
    
    film0_slab_op_fw = Firework([
    WriteVaspFromIOSet(vasp_input_set=ITRelaxSet, structure=get_initial_film(interface)),
    RunVaspCustodian(),
    VaspToDb()], name=f"Optimization", spec = spec)
    
    static_fw = Firework(
        tasks=[
            WriteVaspStaticFromPrev(prev_calc_dir=".", vasp_input_set=ITStaticSet),
            RunVaspCustodian(),
            VaspToDb()
        ],
        spec = spec,
        name = "Static Calculation"
        )
    
    film_slab_workflow = Workflow([film_slab_op_fw, static_fw], name = f"{project_name}:{unique_match_idx}:{interface.interface_properties['termination']}:FilmSlab")
    film_slab_workflow = Workflow([substrate_slab_op_fw, static_fw], name = f"{project_name}:{unique_match_idx}:{interface.interface_properties['termination']}:SubstrateSlab")
    film_slab_workflow = Workflow([film0_slab_op_fw, static_fw], name = f"{project_name}:{unique_match_idx}:{interface.interface_properties['termination']}:Film0Slab")
    return [film_slab_workflow, film_slab_workflow, film_slab_workflow]


def get_static_fw(spec):
    return Firework(
        tasks=[
            WriteVaspStaticFromPrev(prev_calc_dir=".", vasp_input_set=ITStaticSet),
            RunVaspCustodian(),
            VaspToDb()
        ],
        spec = spec,
        name = "Static Calculation"
        )

def LatticeRelaxWF(film_path, substrate_path, project_name, NCORE, db_file, vasp_cmd):
    mopath = os.path.join(os.getcwd(), 'lattices')
    try:
        shutil.rmtree(mopath)
    except:
        print('no folder')
    stcts = {}
    stcts['film'] = Structure.from_file(film_path)
    stcts['substrate'] = Structure.from_file(substrate_path)
    wf = []
    for i in ['film', 'substrate']:
        wf_h = Firework(
            tasks=[
                WriteVaspFromIOSet(vasp_input_set=CstRelaxSet(structure = stcts[i], NCORE=NCORE), structure = stcts[i]),
                RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False),
                VaspToDb(db_file = db_file, additional_fields = {'project_name': project_name, 'it': f'{i}'})
            ], name = f'{project_name}_lattice', spec={"_launch_dir": os.path.join(mopath,f'{i}')})
        wf.append(wf_h)
    wf = Workflow(wf)
    wf.name = project_name
    return wf

def RegistrationScan(cib, project_name, xyzs, termination, slab_length, vacuum_over_film, c_periodic, NCORE, db_file, vasp_cmd):
    wf = []
    count = 0
    mopath = os.path.join(os.getcwd(), project_name)
    try:
        shutil.rmtree(mopath)
    except:
        print('no folder')
    os.mkdir(mopath)
    for i in xyzs:
        x, y, z = i
        if c_periodic:
            vacuum_over_film = gap = z
        else:
            vacuum_over_film = vacuum_over_film
            gap = z
        interface_here = get_one_interface(cib, termination, slab_length, i, vacuum_over_film, c_periodic)
        vacuum_translation = TranslateSitesTransformation(arange(len(interface_here)), [0,0,-vacuum_over_film/interface_here.lattice.c/2+0.01])
        interface_here = vacuum_translation.apply_transformation(interface_here)
        #non-dipole correction
        fw1 = Firework(
        tasks=[
            WriteVaspFromIOSet(vasp_input_set=ITStaticSet(structure = interface_here, NCORE=NCORE, LDIPOL = False, EDIFF = 1e-3), structure = interface_here),
            RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False)
        ], name = f'{project_name}_NDP', spec={"_launch_dir": os.path.join(mopath,str(count))})
        #dipole correction
        fw2 = Firework(
        tasks=[
            ModifyIncar(incar_update = {"LDIPOL": True, "IDIPOL": 3, "LWAVE":False, "EDIFF":1e-5}),
            RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False),
            VaspToDb(db_file = db_file, additional_fields = {'registration_id': count, 'project_name': project_name})
        ], name = f'{project_name}_DP', parents = fw1, spec={"_launch_dir": os.path.join(mopath,str(count))})
        wf.append(fw1)
        wf.append(fw2)
        count += 1
    wf = Workflow(wf)
    wf.name = project_name
    return wf

def HighScoreItWorkflow(ISRker, project_name, NCORE, db_file, vasp_cmd):
    wf = []
    mopath = os.path.join(os.getcwd(), project_name)
    try:
        shutil.rmtree(mopath)
    except:
        print('no folder')
    os.mkdir(mopath)
    with open('ranking_dict.pkl','wb') as f:
        pickle.dump(ISRker.opt_info_dict, f)
    for i in list(ISRker.opt_info_dict):
        rg_id = 0
        for j in ISRker.opt_info_dict[i]['registration_input']:
            cib = CoherentInterfaceBuilder(film_structure=ISRker.film,
                                       substrate_structure=ISRker.substrate,
                                       film_miller=ISRker.unique_matches[i[0]].film_miller,
                                       substrate_miller=ISRker.unique_matches[i[0]].substrate_miller,
                                       zslgen=SubstrateAnalyzer(max_area=30),
                                       termination_ftol=ISRker.termination_ftol,
                                       label_index=True,
                                       filter_out_sym_slabs=False)
            cib.zsl_matches = [ISRker.unique_matches[i[0]]]
            x, y, z = j
            if ISRker.c_periodic:
                vacuum_over_film = gap = z
            else:
                vacuum_over_film = vacuum_over_film
                gap = z
            interface_here = list(cib.get_interfaces(termination=ISRker.opt_info_dict[i]['termination'],
                                       substrate_thickness=ISRker.slab_length,
                                       film_thickness=ISRker.slab_length,
                                       vacuum_over_film=vacuum_over_film,
                                       gap=gap,
                                       in_layers=False))[0]
            CNID = calculate_cnid_in_supercell(interface_here)[0]
            CNID_translation = TranslateSitesTransformation(interface_here.film_indices, x*CNID[:,0] + y*CNID[:,1])
            
            #non-dipole
            fw1 = Firework(
            tasks=[
                WriteVaspFromIOSet(vasp_input_set=ITStaticSet(structure = interface_here, NCORE=NCORE, LDIPOL = False), structure = interface_here),
                RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False)
            ], name = f'high_scores_{i}_{j}', spec={"_launch_dir": os.path.join(mopath,f'{i}_{j}')})
            #dipole correction
            fw2 = Firework(
            tasks=[
                ModifyIncar(incar_update = {"LDIPOL": True, "IDIPOL": 3, "LWAVE":False}),
                RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False),
                VaspToDb(db_file = db_file, additional_fields = {'area': ISRker.areas[i[0]], 'match_id': i[0], 'termination_id': i[1], 'rg_id': rg_id, 'project_name': project_name})
            ], name = f'high_scores_{i}_{j}', parents = fw1, spec={"_launch_dir": os.path.join(mopath,f'{i}_{j}')})
            wf.append(fw1)
            wf.append(fw2)
            rg_id += 1
    wf = Workflow(wf)
    wf.name = project_name
    return wf
