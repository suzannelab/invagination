"""
Delamination behavior in a 2.5D sheet
"""

import random
import numpy as np
import pandas as pd

from tyssue import SheetGeometry
from tyssue.behaviors.sheet_events import contract, ab_pull, type1_at_shorter, type3


def delamination(
    sheet,
    manager,
    face_id,
    contract_rate=2,
    critical_area=1e-2,
    radial_tension=1.0,
    nb_iteration=0,
    nb_iteration_max=20,
    contract_neighbors=True,
    critical_area_neighbors=10,
    contract_span=2,
    geom=SheetGeometry,
    contraction_column="contractility",
):
    """Delamination process
    This function corresponds to the process called "apical constriction"
    in the manuscript
    The cell undergoing delamination first contracts its apical
    area until it reaches a critical area, at which point it starts
    undergoing rearangements with its neighbors, performing
    successive type 1 transitions until the face has only 3 sides,
    when it disepears.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    face_id : int
       the Id of the face undergoing delamination.
    contract_rate : float, default 2
       rate of increase of the face contractility.
    critical_area : float, default 1e-2
       face's area under which the cell starts loosing sides.
    radial_tension : float, default 1.
       tension applied on the face vertices along the
       apical-basal axis.
    nb_iteration : int, default 0
       number of extra iterations where the apical-basal force is applied
       between each type 1 transition
    contract_neighbors : bool, default `False`
       if True, the face contraction triggers contraction of the neighbor
       faces.
    contract_span : int, default 2
       rank of neighbors contracting if contract_neighbor is True. Contraction
       rate for the neighbors is equal to `contract_rate` devided by
       the rank.
    """
    settings = {
        "contract_rate": contract_rate,
        "critical_area": critical_area,
        "radial_tension": radial_tension,
        "nb_iteration": nb_iteration,
        "nb_iteration_max": nb_iteration_max,
        "contract_neighbors": contract_neighbors,
        "critical_area_neighbors": critical_area_neighbors,
        "contract_span": contract_span,
        "geom": geom,
        "contraction_column": contraction_column,
    }

    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return

    if sheet.face_df.loc[face, "is_relaxation"]:
        relaxation(sheet, face, contract_rate)

    face_area = sheet.face_df.loc[face, "area"]
    num_sides = sheet.face_df.loc[face, "num_sides"]

    if face_area > critical_area:
        contract(sheet, face, contract_rate, True)

        if contract_neighbors & (face_area < critical_area_neighbors):
            neighbors = sheet.get_neighborhood(face, contract_span).dropna()
            neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values

            manager.extend([
                (contraction_neighbours, neighbor['id'],
                 (contract_rate ** (1 / 2 ** neighbor['order']),
                  critical_area, 50))  # TODO: check this
                for _, neighbor in neighbors.iterrows()
                ])
        done = False

    elif face_area <= critical_area:
        if nb_iteration < nb_iteration_max:
            settings["nb_iteration"] = nb_iteration + 1
            ab_pull(sheet, face, radial_tension, True)
            done = False
        elif nb_iteration >= nb_iteration_max:
            if num_sides > 3:
                type1_at_shorter(sheet, face, geom, False)
                done = False
            elif num_sides <= 3:
                type3(sheet, face, geom)
                done = True

    if not done:
        manager.append(delamination, face_id, kwargs=settings)


def relaxation(sheet, face, contractility_decrease, contraction_column="contractility"):

    initial_contractility = 1.12
    new_contractility = (
        sheet.face_df.loc[face, contraction_column] / contractility_decrease
    )

    if new_contractility >= (initial_contractility / 2):
        sheet.face_df.loc[face, contraction_column] = new_contractility
        sheet.face_df.loc[face, "prefered_area"] *= contractility_decrease


def neighbors_contraction(
    sheet,
    manager,
    face_id,
    contractile_increase=1.0,
    critical_area=1e-2,
    max_contractility=10,
    contraction_column="contractility",
):
    """Custom single step contraction event.
    """
    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return
    if (sheet.face_df.loc[face, "area"] < critical_area) or (
        sheet.face_df.loc[face, contraction_column] > max_contractility
    ):
        return
    contract(sheet, face, contractile_increase, True)


def type1_transition(sheet, manager, face_id, critical_length=0.3, geom=SheetGeometry):
    """Custom type 1 transition event that tests if
    the the shorter edge of the face is smaller than
    the critical length.
    """
    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    if min(edges["length"]) < critical_length:
        type1_at_shorter(sheet, face, geom)


def face_elimination(sheet, manager, face_id, geom=SheetGeometry):
    """Removes the face with if face_id from the sheet
    """
    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return
    type3(sheet, face, geom)


def check_tri_faces(sheet, manager):
    """Three neighbourghs cell elimination
    Add all cells with three neighbourghs in the manager
    to be eliminated at the next time step.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    """

    tri_faces = sheet.face_df[
        (sheet.face_df["num_sides"] < 4) & (sheet.face_df["is_mesoderm"] is False)
    ]["id"]
    manager.extend(
        [
            (face_elimination, f, (), {"geom": sheet.settings["delamination"]["geom"]})
            for f in tri_faces
        ]
    )


def check_enter_in_process(
    sheet, manager, mesoderm, t=1, base=0.003, amp=0.5, largeur=2.5, density_proba=6
):
    """
    Check if face in mesoderm can enter in the process.
    If cell enter in the process, it will be removed from list_cell
    and add to the manager with it's process.
    """
    for f in mesoderm:
        if enter_in_process(sheet, f, base, amp, largeur, density_proba):
            manager.append(delamination, f, kwargs=sheet.settings["delamination"])


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


def define_mesoderm(sheet, a=145.0, b=40.0, coords=["x", "z", "y"]):
    """
    Define an oval area that will become the mesoderm.
    a: radius on the first-axis
    b: radius on the second-axis
    """
    x, y, z = coords
    x_ = sheet.face_df[x] / b
    y_ = sheet.face_df[y] / a

    radius = x_ ** 2 + y_ ** 2
    height = sheet.face_df[z]

    sheet.face_df["is_mesoderm"] = (radius <= 1) & (height > 0)

    # Remove not active face
    face_not_active = sheet.edge_df[~sheet.edge_df.is_active.astype(bool)].face.unique()
    sheet.face_df.loc[face_not_active, "is_mesoderm"] = False
