"""
Delamination behavior in a 2.5D sheet
"""

import random
import numpy as np

from tyssue.behaviors.sheet.delamination_events import (delamination, constriction)


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
            manager.append(constriction, face_id=f, **sheet.settings["delamination"])


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
