"""
Contractility gradient behavior in a 2.5D sheet

"""

import random
import numpy as np

from tyssue import SheetGeometry
from tyssue.behaviors.sheet_events import (contract,
                                           ab_pull)
from invagination.delamination import contraction_neighbours


def constriction(sheet, manager, face_id,
                 contract_rate=2,
                 critical_area=1e-2,
                 radial_tension=1.,
                 nb_iteration=0,
                 nb_iteration_max=20,
                 contract_neighbors=True,
                 critical_area_neighbors=10,
                 contract_span=2,
                 basal_contract_rate=1.001,
                 geom=SheetGeometry,
                 can_pull=True):

    face = sheet.idx_lookup(face_id, 'face')
    if face is None:
        return

    if sheet.face_df.loc[face, 'is_relaxation']:
        relaxation(sheet, face, contract_rate)

    else:
        face_area = sheet.face_df.loc[face, 'area']

        if face_area > critical_area:
            contract(sheet, face, contract_rate, True)
            # increase_linear_tension(sheet, face, contract_rate)

            if (contract_neighbors & (face_area < critical_area_neighbors)):
                neighbors = sheet.get_neighborhood(
                    face, contract_span).dropna()
                neighbors['id'] = sheet.face_df.loc[
                    neighbors.face, 'id'].values

                manager.extend([
                    (contraction_neighbours, neighbor['id'],
                     (-((basal_contract_rate - contract_rate) / contract_span) *
                      neighbor['order'] + contract_rate,
                      critical_area, 50))  # TODO: check this
                    for _, neighbor in neighbors.iterrows()
                ])
        if can_pull:
            proba_tension = np.exp(-face_area / critical_area)
            aleatory_number = random.uniform(0, 1)
            if aleatory_number < proba_tension:
                ab_pull(sheet, face, radial_tension, True)

    manager.append(constriction, face_id,
                   (contract_rate,
                    sheet.settings['delamination']['critical_area'],
                    sheet.settings['delamination']['radial_tension'],
                    sheet.settings['delamination']['nb_iteration'],
                    sheet.settings['delamination']['nb_iteration_max'],
                    sheet.settings['delamination']['contract_neighbors'],
                    sheet.settings['delamination'][
                        'critical_area_neighbors'],
                    sheet.settings['delamination']['contract_span'],
                    sheet.settings['delamination']['basal_contract_rate'],
                    sheet.settings['delamination']['geom'],
                    sheet.settings['delamination']['can_pull']
                    ))


def relaxation(sheet, face, contractility_decrease):

    initial_contractility = 1.12
    new_contractility = sheet.face_df.loc[
        face, 'contractility'] / contractility_decrease

    if new_contractility >= (initial_contractility / 2):
        sheet.face_df.loc[face, 'contractility'] = new_contractility
        sheet.face_df.loc[face, 'prefered_area'] *= contractility_decrease


def increase_linear_tension(sheet, face, line_tension, geom=SheetGeometry):
    edges = sheet.edge_df[sheet.edge_df['face'] == face]
    for index, edge in edges.iterrows():
        angle_ = np.arctan2(sheet.edge_df.dx, sheet.edge_df.dy)

        if np.abs(angle_) < np.pi / 4:
            sheet.edge_df.loc[edge.name, 'line_tension'] *= line_tension
