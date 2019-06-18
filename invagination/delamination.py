"""
Delamination behavior in a 2.5D sheet
"""

import os
import random
import numpy as np

from invagination.ellipsoid import RadialTension, VitellineElasticity

from tyssue.io import hdf5
from tyssue.dynamics import effectors
from tyssue.dynamics.factory import model_factory
from tyssue.geometry.sheet_geometry import EllipsoidGeometry as geom
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import check_tri_faces
from tyssue.behaviors.sheet.delamination_events import constriction


model = model_factory(
    [
        RadialTension,
        VitellineElasticity,
        effectors.FaceContractility,
        effectors.FaceAreaElasticity,
        # effectors.CellVolumeElasticity,
        effectors.LumenVolumeElasticity,
    ],
    effectors.FaceAreaElasticity,
)


def constriction_rate(x, max_constriction_rate, k, w):
    """Calculate constriction rate of a cell according to its position in the tissue.
    Parameters
    ----------
    x: position of cell in the tissue
    max_constriction: maximal constriction cell for all cell in the tissue.
    k: steepness coefficient characterizing the profile decay
    w: width of the profile
    """

    c_rate = 1 + (max_constriction_rate - 1) * (
        (1 + np.exp(-k * w)) / (1 + np.exp(k * (abs(x) - w)))
    )

    return c_rate


def apical_cut(sheet, pos_z):
    """
    Define cell which are cut to simulate apical ablation.
    Parameters
    ----------
    sheet:
    pos_z: position where mesodermal cell are "cut"
    """

    srce_z = sheet.upcast_srce(sheet.vert_df["z"])
    trgt_z = sheet.upcast_trgt(sheet.vert_df["z"])
    srce_y = sheet.upcast_srce(sheet.vert_df["y"])
    trgt_y = sheet.upcast_trgt(sheet.vert_df["y"])

    meso_edge = sheet.upcast_face(sheet.face_df["is_mesoderm"])
    cut_edges = sheet.edge_df[
        (
            ((srce_z < pos_z) & (trgt_z >= pos_z))
            | ((srce_z >= pos_z) & (trgt_z <= pos_z))
        )
        & meso_edge
        & ((srce_y > 0) & (trgt_y > 0))
    ]

    cut_faces = cut_edges["face"].unique()

    sheet.face_df.loc[cut_faces, ["contractility", "area_elasticity"]] = 1e-2

    sheet.face_df.loc[cut_faces, "is_mesoderm"] = False


def delamination_process(
    sim_save_dir,
    sheet,
    max_contractility_rate,
    critical_area,
    radial_tension,
    nb_iteraction_max,
    profile_width,
    k,
    iteration,
    cable_cut=False,
    apical_cut=False,
    nb_apical_cut=2,
):

    """
    Initiate simulation before running according to parameters.
    Parameters
    ----------
    dirname: saving directory to hf5 file
    sheet:
    max_contractility_rate: maximal constriction cell for all cell in the tissue.
    k: steepness coefficient characterizing the profile decay
    profile_width: width of the profile
    cable_cut: True/False, define if apico-basal force is exerted
    apical_cut: True/False, define if a domain of cell is isolated
    nb_apical_cut: Define number of apical cut (1 or 2)
    """

    # Directory definition
    dirname = "{}_contractility_{}_critical_area_{}_radialtension_{}".format(
        max_contractility_rate, critical_area, radial_tension, iteration
    )
    dirname = os.path.join(sim_save_dir, dirname)

    print("starting {}".format(dirname))
    try:
        os.mkdir(dirname)
    except IOError:
        pass

    settings = {
        "critical_area": critical_area,
        "radial_tension": radial_tension,
        "nb_iteration": 0,
        "nb_iteration_max": nb_iteraction_max,
        "contract_neighbors": True,
        "critical_area_neighbors": 12,
        "contract_span": 3,
        "basal_contract_rate": 1.001,
        "geom": geom,
        "contraction_column": "contractility",
    }

    # Add some information to the sheet
    sheet2 = sheet.copy(deep_copy=True)
    sheet2.face_df["id"] = sheet2.face_df.index.values
    sheet2.settings["delamination"] = settings
    settings2 = {"critical_length": 0.3}
    sheet2.settings["T1"] = settings2

    # """ Initiale find minimal energy
    # To be sure we are at the equilibrium
    solver = Solver
    solver_kw = {
        "minimize": {"method": "L-BFGS-B", "options": {"ftol": 1e-8, "gtol": 1e-8}}
    }
    res = solver.find_energy_min(sheet2, geom, model, **solver_kw)

    sheet2 = run_sim(
        dirname,
        solver,
        solver_kw,
        sheet2,
        geom,
        model,
        max_contractility_rate,
        profile_width,
        k,
        cable_cut,
        apical_cut,
        nb_apical_cut,
    )

    print("{} done".format(dirname))
    print("~~~~~~~~~~~~~~~~~~~~~\n")


def run_sim(
    dirname,
    solver,
    solver_kw,
    sheet,
    geom,
    model,
    max_contractility_rate,
    profile_width,
    k,
    cable_cut=False,
    apical_cut=False,
    nb_apical_cut=2,
):
    """
    Run simulation according to parameters.
    Parameters
    ----------
    dirname: saving directory to hf5 file
    solver:
    solver_kw: solver arguments
    sheet:
    geom:
    model:
    max_contractility_rate: maximal constriction cell for all cell in the tissue.
    k: steepness coefficient characterizing the profile decay
    profile_width: width of the profile
    cable_cut: True/False, define if apico-basal force is exerted
    apical_cut: True/False, define if a domain of cell is isolated
    nb_apical_cut: Define number of apical cut (1 or 2)
    """

    # Initiate manager
    manager = EventManager("face")
    sheet.face_df["enter_in_process"] = 0

    t = 0
    stop = 200
    sheet.face_df["contract_rate"] = 0

    if apical_cut:
        if nb_apical_cut == 1:
            # posterior apical ablation
            apical_cut(sheet, 45.0)

        elif nb_apical_cut == 2:
            # anterior & posterior apical ablation
            apical_cut(sheet, 45.0)
            apical_cut(sheet, -45.0)

    # Add all cells in constriction process
    for f in sheet.face_df[sheet.face_df["is_mesoderm"]].index:
        x = sheet.face_df.loc[f, "x"]
        c_rate = constriction_rate(x, max_contractility_rate, k, profile_width)

        sheet.face_df.loc[f, "contract_rate"] = c_rate
        delam_kwargs = sheet.settings["delamination"].copy()
        delam_kwargs.update(
            {
                "face_id": f,
                "contract_rate": c_rate,
                "current_traction": 0,
                "max_traction": 30,
            }
        )
        manager.append(constriction, **delam_kwargs)

    for f in sheet.face_df[sheet.face_df["is_relaxation"]].index:
        delam_kwargs = sheet.settings["delamination"].copy()
        delam_kwargs.update(
            {
                "face_id": f,
                "contract_rate": max_contractility_rate,
                "current_traction": 0,
                "max_traction": 30,
            }
        )
        manager.append(constriction, **delam_kwargs)

    while manager.current and t < stop:
        # Clean radial tension on all vertices
        sheet.vert_df["radial_tension"] = 0
        manager.execute(sheet)

        if cable_cut:
            # Mettre ici la mise à 0 de la force AB dans la zone -45 45
            sheet.vert_df["radial_tension"] = [
                0 if ((z > -45.0) and (z < 45.0)) else rad
                for (z, rad) in sheet.vert_df[["z", "radial_tension"]].values
            ]

        res = solver.find_energy_min(sheet, geom, model, **solver_kw)

        # add noise on vertex position to avoid local minimal.
        sheet.vert_df[["x", "y"]] += np.random.normal(scale=1e-3, size=(sheet.Nv, 2))
        geom.update_all(sheet)

        # Save result in each time step.
        figname = os.path.join(dirname, "invagination_{:04d}.png".format(t))
        hdfname = figname[:-3] + "hf5"
        hdf5.save_datasets(hdfname, sheet)

        # Add cells with initially 3 neighbourghs to be eliminated.
        check_tri_faces(sheet, manager)

        manager.update()
        t += 1

    return sheet
