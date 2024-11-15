from fireworks import FireTaskBase, FWAction, explicit_serialize, Firework, Workflow
from atomate.vasp.firetasks.run_calc import RunVaspCustodian
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet, ModifyIncar
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.analysis.interfaces.coherent_interfaces import get_rot_3d_for_2d, CoherentInterfaceBuilder
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from scipy.linalg import polar
from InterOptimus.CNID import calculate_cnid_in_supercell
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from numpy import arange, ceil, savetxt, dot, meshgrid, array, inf
from numpy.linalg import norm, inv
import pickle
import shutil
import os
from InterOptimus.tool import get_one_interface, trans_to_bottom, add_sele_dyn, add_sele_dyn_slab

def get_potcar_dict():
    return {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 'Ar': 'Ar', 'As': 'As', 'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be_sv', 'Bi': 'Bi', 'Br': 'Br', 'C': 'C', 'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 'Cl': 'Cl', 'Co': 'Co', 'Cr': 'Cr_pv', 'Cs': 'Cs_sv', 'Cu': 'Cu_pv', 'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu', 'F': 'F', 'Fe': 'Fe_pv', 'Ga': 'Ga_d', 'Gd': 'Gd', 'Ge': 'Ge_d', 'H': 'H', 'He': 'He', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv', 'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 'Mg': 'Mg_pv', 'Mn': 'Mn', 'Mo': 'Mo_pv', 'N': 'N', 'Na': 'Na_pv', 'Nb': 'Nb_pv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Ni': 'Ni', 'Np': 'Np', 'O': 'O', 'Os': 'Os_pv', 'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd', 'Pm': 'Pm_3', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'Rb': 'Rb_sv', 'Re': 'Re_pv', 'Rh': 'Rh_pv', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 'Ti': 'Ti_pv', 'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 'V': 'V_pv', 'W': 'W_pv', 'Xe': 'Xe', 'Y': 'Y_sv', 'Yb': 'Yb_2', 'Zn': 'Zn', 'Zr': 'Zr_sv'}

def get_potcar(structure):
    return Potcar([get_potcar_dict()[i.symbol] for i in structure.elements], functional = 'PBE_54')

def CstRelaxSet(structure, ENCUT_scale = 1, NCORE = 12, Kdense = 500):
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
    
def get_initial_film(interface, match):
    """
    get the non-deformed film
    """
    sub_vs0 = interface.interface_properties['substrate_sl_vectors']
    film_vs0 = interface.interface_properties['film_sl_vectors']
    sub_vs1 = interface.lattice.matrix[:2]
    R1 = get_rot_3d_for_2d(sub_vs0, sub_vs1)
    original_trans = match.match_transformation
    R0, T = polar(original_trans)
    R = dot(R1, R0)
    strain_inv = dot(dot(R, inv(T)), inv(R))
    #print(strain_inv)
    new_lattice = dot(strain_inv, interface.lattice.matrix.T).T
    return Structure(new_lattice, interface.film.species, interface.film.frac_coords)

def NDPDPWF(structure, project_name, NCORE, db_file, vasp_cmd, additional_fields, spec, incar_update):
    fw1 = Firework(
    tasks=[
        WriteVaspFromIOSet(vasp_input_set=ITStaticSet(structure = structure, NCORE = NCORE, LDIPOL = False, EDIFF = 1e-4), structure = structure),
        RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False)
    ], name = f'{project_name}_NDP', spec=spec)
    #dipole correction
    fw2 = Firework(
    tasks=[
        ModifyIncar(incar_update = incar_update),
        RunVaspCustodian(vasp_cmd = vasp_cmd, handler_group = "no_handler", gzip_output = False),
        VaspToDb(db_file = db_file, additional_fields = additional_fields)
    ], name = f'{project_name}_DP', parents = fw1, spec=spec)
    return fw1, fw2

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

def readDBvasp(db, field):
    data = db.collection.find_one(field)
    energy = inf
    if data != None:
        if data['state'] == 'successful':
            energy = data['output']['energy']
    return energy

def SlabEnergyWorkflows(interface, match, project_name, NCORE, db_file, vasp_cmd, relax = False, calc_initial_film = False, tag = '', IOPTCELL = None, ISIF = 2):
    """
    work flow to calculate slab energy
    """
    if project_name not in os.listdir():
        os.mkdir(project_name)
    mopath = os.path.join(os.getcwd(), project_name)
    film_slab = trans_to_bottom(interface.film)
    substrate_slab = add_sele_dyn_slab(interface.substrate)
    film0_slab = trans_to_bottom(get_initial_film(interface, match))
    wf = []
    if relax:
        incar_update = {"EDIFFG": -0.05, "IOPTCELL": IOPTCELL, "ISIF": ISIF, "NSW": 300, \
                        "LDIPOL": True, "IDIPOL": 3, "EDIFF": 1e-5}
    else:
        incar_update = {"LDIPOL": True, "IDIPOL": 3, "LWAVE": False, "EDIFF": 1e-5}
    fw1, fw2 = NDPDPWF(film_slab, project_name, NCORE, db_file, vasp_cmd, \
                             {'project_name': project_name, 'job': f'film_t{tag}'}, \
                             {"_launch_dir": os.path.join(mopath, f'film_t{tag}'), 'job': f'film_t{tag}'}, \
                            incar_update)
    wf.append(fw1)
    wf.append(fw2)
    
    fw1, fw2 = NDPDPWF(substrate_slab, project_name, NCORE, db_file, vasp_cmd, \
                             {'project_name': project_name, 'job': f'substrate{tag}'}, \
                             {"_launch_dir": os.path.join(mopath, f'substrate{tag}'), 'job': f'substrate{tag}'}, \
                            incar_update)
    wf.append(fw1)
    wf.append(fw2)
    
    if calc_initial_film:
        fw1, fw2 = NDPDPWF(film0_slab, project_name, NCORE, db_file, vasp_cmd, \
                                 {'project_name': project_name, 'job': f'film_0{tag}'}, \
                                 {"_launch_dir": os.path.join(mopath, f'film_t{tag}'), 'job': f'film_0{tag}'}, \
                                incar_update)
        wf.append(fw1)
        wf.append(fw2)
    return wf

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
            VaspToDb(db_file = db_file, additional_fields = {'job': f'rg_{count}', 'project_name': project_name})
        ], name = f'{project_name}_DP', parents = fw1, spec={"_launch_dir": os.path.join(mopath,str(count))})
        wf.append(fw1)
        wf.append(fw2)
        count += 1
    se_wf = SlabEnergyWorkflows(interface_here, cib.zsl_matches[0], project_name, NCORE, db_file, vasp_cmd)
    wf = Workflow(wf+se_wf)
    wf.name = project_name
    return wf

def ScoreRankerWF(ISRker, selected_its, project_name, NCORE, db_file, vasp_cmd):
    count = 0
    mopath = os.path.join(os.getcwd(), project_name)
    try:
        shutil.rmtree(mopath)
    except:
        print('no folder')
    os.mkdir(mopath)
    if ISRker.c_periodic:
        IOPTCELL = "0 0 0 0 0 0 0 0 1"
        ISIF = 3
    else:
        IOPTCELL = None
        ISIF = 2
    existing_pairs = []
    with open(f'{mopath}/HTPT.dat', 'w') as f:
        for i in selected_its:
            f.write(f'{i[0][0]} {i[0][1]} {i[1][0]} {i[1][1]} {i[1][2]} {i[-1]}\n')
    wf = []
    for i in selected_its:
        cib = CoherentInterfaceBuilder(film_structure=ISRker.film,
                           substrate_structure=ISRker.substrate,
                           film_miller=ISRker.unique_matches[i[0][0]].film_miller,
                           substrate_miller=ISRker.unique_matches[i[0][0]].substrate_miller,
                           zslgen=SubstrateAnalyzer(max_area=30),termination_ftol=ISRker.termination_ftol,
                           label_index=True,
                           filter_out_sym_slabs=False)
        cib.zsl_matches = [ISRker.unique_matches[i[0][0]]]
        interface_here = get_one_interface(cib, i[2], ISRker.slab_length, i[1], ISRker.vacuum_over_film, ISRker.c_periodic)
        interface_here = trans_to_bottom(interface_here)
        interface_here = add_sele_dyn(interface_here)
        incar_update = {"EDIFFG": -0.05, "IOPTCELL": IOPTCELL, "ISIF": ISIF, "NSW": 300, \
                        "LDIPOL": True, "IDIPOL": 3, "EDIFF": 1e-5}
        fw1, fw2 = NDPDPWF(interface_here, project_name, NCORE, db_file, vasp_cmd, \
                             {'project_name': project_name, 'job': f'it_{count}'}, \
                             {"_launch_dir": os.path.join(mopath, f'it_{count}'), 'job': f'it_{count}'}, \
                            incar_update)
        wf.append(fw1)
        wf.append(fw2)
        count += 1
        if i[0] not in existing_pairs:
            fw_se = SlabEnergyWorkflows(interface_here, ISRker.unique_matches[i[0][0]], project_name, NCORE, db_file, vasp_cmd, relax = True, calc_initial_film = False, tag = f'_{i[0][0]}_{i[0][1]}', IOPTCELL = IOPTCELL, ISIF = ISIF)
            wf += fw_se
            existing_pairs.append(i[0])
    wf = Workflow(wf)
    wf.name = project_name
    return wf
        

