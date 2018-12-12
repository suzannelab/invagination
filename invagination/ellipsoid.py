import numpy as np
import pandas as pd

from tyssue import SheetGeometry, BulkGeometry, ClosedMonolayerGeometry
from tyssue.utils.utils import _to_3d, to_nd
from tyssue.dynamics.sheet_gradients import height_grad
from tyssue.dynamics import units, effectors, model_factory


class EllipsoidGeometry(SheetGeometry):
    """
    x = a \cos(\phi)

    """

    @staticmethod
    def update_height(eptm):
        """see data/svg/ellipsoid_geometry.svg
        """

        a, b, c = eptm.settings['abc']
        vitelline_space = eptm.settings['vitelline_space']
        eptm.vert_df['rho'] = np.linalg.norm(eptm.vert_df[['x', 'y']], axis=1)
        eptm.vert_df['theta'] = np.arcsin((eptm.vert_df.z / c).clip(-1, 1))
        eptm.vert_df['phi'] = np.arctan2(eptm.vert_df.y, eptm.vert_df.x)
        eptm.vert_df['vitelline_rho'] = (
            a + vitelline_space) * np.cos(eptm.vert_df['theta'])

        eptm.vert_df['basal_shift'] = (eptm.vert_df['vitelline_rho']
            - vitelline_space - a + eptm.specs['vert']['basal_shift'])

        eptm.vert_df['delta_rho'] = eptm.vert_df.eval(
            'rho - vitelline_rho').clip(lower=0)
        SheetGeometry.update_height(eptm)

    @staticmethod
    def update_vol(eptm):

        if "segment" in eptm.edge_df:
            for c in 'xyz':
                eptm.edge_df['c' + c] = 0
            ClosedMonolayerGeometry.update_vol(eptm)
        ClosedMonolayerGeometry.update_lumen_vol(eptm)

    @staticmethod
    def scale(eptm, scale, coords):
        SheetGeometry.scale(eptm, scale, coords)
        eptm.settings['abc'] = [u * scale for u in eptm.settings['abc']]


class VitellineElasticity(effectors.AbstractEffector):

    dimensions = units.line_elasticity
    magnitude = 'vitelline_K'
    label = 'Vitelline elasticity'
    element = 'vert'
    specs = {
        'vert': {
            'vitelline_K',
            'is_active',
            'delta_rho'}
        }  # distance to the vitelline membrane

    @staticmethod
    def energy(eptm):
        return eptm.vert_df.eval(
            'delta_rho**2 * vitelline_K/2')

    @staticmethod
    def gradient(eptm):
        grad = height_grad(eptm) * _to_3d(
            eptm.vert_df.eval('vitelline_K * delta_rho'))
        grad.columns = ['g' + c for c in eptm.coords]
        return grad, None


class RadialTension(effectors.AbstractEffector):

    dimensions = units.line_tension
    magnitude = 'radial_tension'
    label = 'Apical basal tension'
    element = 'vert'
    specs = {'vert':{'is_active',
             'height',
             'radial_tension'}}

    @staticmethod
    def energy(eptm):
        return eptm.vert_df.eval(
            'height * radial_tension * is_active')

    @staticmethod
    def gradient(eptm):
        grad = height_grad(eptm) * to_nd(
            eptm.vert_df.eval('radial_tension'), 3)
        grad.columns = ['g'+c for c in eptm.coords]
        return grad, None


model = model_factory(
    [
    RadialTension,
    VitellineElasticity,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity,
    #effectors.CellVolumeElasticity,
    effectors.LumenVolumeElasticity,
    ], effectors.FaceAreaElasticity)
