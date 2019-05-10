"""
Delamination behavior in a 2.5D sheet
"""

import os
import random
import numpy as np

from invagination.ellipsoid import (RadialTension,
                                    VitellineElasticity)

from tyssue.io import hdf5
from tyssue.dynamics import effectors
from tyssue.dynamics.factory import model_factory
from tyssue.geometry.sheet_geometry import EllipsoidGeometry as geom
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import check_tri_faces
from tyssue.behaviors.sheet.delamination_events import (constriction)


def check_enter_in_process(
    sheet, manager, mesoderm, base=0.003, amp=0.5, largeur=2.5, density_proba=6
):
    """
    Check if face in mesoderm can enter in the process.
    If cell enter in the process, it will be removed from list_cell
    and add to the manager with it's process.
    """
    for f in mesoderm:
        if enter_in_process(sheet, f, base, amp, largeur, density_proba):
            manager.append(constriction, face_id=f, **
                           sheet.settings["delamination"])


def gaussian(x, base=0, amp=0.8, width=0.7, n=2):
    """Gaussian or Gaussian like function
    Parameters
    ----------
    """
    # xs = sheet.face_df.loc[mesoderm, 'x']
    # w = width * xs.ptp() / 2.
    gauss = base + amp * np.exp(-abs(x) ** n / width ** n)

    return gauss


def enter_in_process(sheet, f, base=0.003, amp=0.5, width=2.5, n=6):
    """
    Define if the face can enter in a process following a gaussian curve.
    """
    face = sheet.idx_lookup(f, "face")
    if face is None:
        return False

    if sheet.face_df.loc[face, "enter_in_process"] == 1:
        return False

    x = sheet.face_df.loc[face, "x"]
    gauss_position = gaussian(x, base, amp, width, n)
    aleatory_number = random.uniform(0, 1)

    if aleatory_number < gauss_position:
        sheet.face_df.loc[face, "enter_in_process"] = 1
        return True
    return False


model = model_factory(
    [
        RadialTension,
        VitellineElasticity,
        effectors.FaceContractility,
        effectors.FaceAreaElasticity,
        # effectors.CellVolumeElasticity,
        effectors.LumenVolumeElasticity,
    ], effectors.FaceAreaElasticity)


def delamination_process(sim_save_dir, sheet, max_contractility_rate, critical_area,
                         radial_tension, nb_iteraction_max,
                         profile_width, k, iteration,
                         cable_cut=False, apical_cut=False, nb_apical_cut=2):
    # Directory definition
    dirname = '{}_contractility_{}_critical_area_{}_radialtension_{}'.format(
        max_contractility_rate, critical_area, radial_tension, iteration)
    dirname = os.path.join(sim_save_dir, dirname)

    print('starting {}'.format(dirname))
    try:
        os.mkdir(dirname)
    except IOError:
        pass

    settings = {'critical_area': critical_area,
                'radial_tension': radial_tension,
                'nb_iteration': 0,
                'nb_iteration_max': nb_iteraction_max,
                'contract_neighbors': True,
                'critical_area_neighbors': 12,
                'contract_span': 3,
                'basal_contract_rate': 1.001,
                'geom': geom,
                'contraction_column': 'contractility'}

    # Add some information to the sheet
    sheet2 = sheet.copy(deep_copy=True)
    sheet2.face_df['id'] = sheet2.face_df.index.values
    sheet2.settings['delamination'] = settings
    settings2 = {'critical_length': 0.3}
    sheet2.settings['T1'] = settings2

    #""" Initiale find minimal energy
    # To be sure we are at the equilibrium
    solver = Solver
    solver_kw = {'minimize': {'method': 'L-BFGS-B',
                              'options': {'ftol': 1e-8,
                                          'gtol': 1e-8}}}
    res = solver.find_energy_min(sheet2, geom, model, **solver_kw)

    sheet2 = run_sim(dirname, solver, solver_kw, sheet2, geom, model,
                     max_contractility_rate, profile_width, k,
                     cable_cut, apical_cut, nb_apical_cut)

    print('{} done'.format(dirname))
    print('~~~~~~~~~~~~~~~~~~~~~\n')


def run_sim(dirname, solver, solver_kw, sheet, geom, model,
            max_contractility_rate, profile_width, k,
            cable_cut=False, apical_cut=False, nb_apical_cut=2):

    # Initiate manager
    manager = EventManager('face')
    sheet.face_df['enter_in_process'] = 0

    t = 0
    stop = 200
    sheet.face_df['contract_rate'] = 0

    if apical_cut:
        if nb_apical_cut == 1:
            # posterior apical ablation
            z_ablat = 45.

            srce_z = sheet.upcast_srce(sheet.vert_df['z'])
            trgt_z = sheet.upcast_trgt(sheet.vert_df['z'])
            srce_y = sheet.upcast_srce(sheet.vert_df['y'])
            trgt_y = sheet.upcast_trgt(sheet.vert_df['y'])

            meso_edge = sheet.upcast_face(sheet.face_df['is_mesoderm'])
            cut_edges = sheet.edge_df[(((srce_z < z_ablat) & (trgt_z >= z_ablat))
                                       | ((srce_z >= z_ablat) & (trgt_z <= z_ablat)))
                                      & meso_edge & ((srce_y > 0) & (trgt_y > 0))]
            cut_faces = cut_edges['face'].unique()
            sheet.face_df.loc[cut_faces, [
                'contractility', 'area_elasticity']] = 1e-2

        elif nb_apical_cut == 2:
            # anterior & posterior apical ablation
            z_ablat_post = 45.
            z_ablat_ant = -45.

            srce_z = sheet.upcast_srce(sheet.vert_df['z'])
            trgt_z = sheet.upcast_trgt(sheet.vert_df['z'])
            srce_y = sheet.upcast_srce(sheet.vert_df['y'])
            trgt_y = sheet.upcast_trgt(sheet.vert_df['y'])

            meso_edge = sheet.upcast_face(sheet.face_df['is_mesoderm'])
            cut_edges = sheet.edge_df[(((srce_z < z_ablat_post) & (trgt_z >= z_ablat_post))
                                       | ((srce_z >= z_ablat_post) & (trgt_z <= z_ablat_post))
                                       | ((srce_z < z_ablat_ant) & (trgt_z >= z_ablat_ant))
                                       | ((srce_z >= z_ablat_ant) & (trgt_z <= z_ablat_ant)))
                                      & meso_edge & ((srce_y > 0) & (trgt_y > 0))]
            cut_faces = cut_edges['face'].unique()
            sheet.face_df.loc[cut_faces, [
                'contractility', 'area_elasticity']] = 1e-2
            sheet.face_df.loc[cut_faces, 'is_mesoderm'] = False

    # Add all cells in constriction process
    for f in sheet.face_df[sheet.face_df['is_mesoderm']].index:
        x = sheet.face_df.loc[f, 'x']
        c_rate = 1 + (max_contractility_rate - 1) * ((1 + np.exp(-k * profile_width)) /
                                                     (1 + np.exp(k * (abs(x) - profile_width))))

        sheet.face_df.loc[f, 'contract_rate'] = c_rate
        delam_kwargs = sheet.settings["delamination"].copy()
        delam_kwargs.update(
            {
                'face_id': f,
                'contract_rate': c_rate,
                'current_traction': 0,
                'max_traction': 30
            }
        )
        manager.append(constriction, **delam_kwargs)

    for f in sheet.face_df[sheet.face_df['is_relaxation']].index:
        delam_kwargs = sheet.settings["delamination"].copy()
        delam_kwargs.update(
            {
                'face_id': f,
                'contract_rate': max_contractility_rate,
                'current_traction': 0,
                'max_traction': 30
            }
        )
        manager.append(constriction, **delam_kwargs)

    while manager.current and t < stop:
        # Clean radial tension on all vertices
        sheet.vert_df['radial_tension'] = 0
        manager.execute(sheet)

        if cable_cut:
            # Mettre ici la mise à 0 de la force AB dans la zone -45 45
            sheet.vert_df['radial_tension'] = [0 if ((z > -45.) and (z < 45.))
                                               else rad
                                               for (z, rad) in sheet.vert_df[['z', 'radial_tension']].values]

        res = solver.find_energy_min(sheet, geom, model, **solver_kw)

        # add noise on vertex position to avoid local minimal.
        sheet.vert_df[
            ['x', 'y']] += np.random.normal(scale=1e-3, size=(sheet.Nv, 2))
        geom.update_all(sheet)

        figname = os.path.join(
            dirname, 'invagination_{:04d}.png'.format(t))
        hdfname = figname[:-3] + 'hf5'
        hdf5.save_datasets(hdfname, sheet)

        # Add cells with initially 3 neighbourghs to be eliminated
        check_tri_faces(sheet, manager)
        # Add T1 transition for face with at least one edge shorter than critical length
        #[manager.append(type1_transition, f, kwargs=sheet.settings['T1']) for f in sheet.edge_df[
        # sheet.edge_df['length'] <
        # sheet.settings['T1']['critical_length']]['face'].unique()]

        manager.update()
        t += 1

    return sheet
