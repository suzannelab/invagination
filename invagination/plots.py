"""
Some basic plot function
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tyssue
from tyssue.draw.plt_draw import quick_edge_draw, sheet_view
from tyssue.config.draw import sheet_spec
from invagination.toolbox import (open_sheet,
                                  force_ratio,
                                  define_time_max_depth)

draw_specs = tyssue.config.draw.sheet_spec()


def mesoderm_position(sheet, mesoderm_cells):
    """
    Plot two pannel view of the sheet and
    the position of the mesoderm cells
    """

    fig, axes = plt.subplots(1, 2, sharey=True)

    fig, ax = quick_edge_draw(sheet, ['z', 'x'],
                              ax=axes[0],
                              alpha=0.7)
    ax.plot(sheet.face_df.loc[mesoderm_cells, 'z'],
            sheet.face_df.loc[mesoderm_cells, 'x'],
            'o', alpha=0.8, ms=5)

    fig, ax = quick_edge_draw(sheet, ['y', 'x'],
                              ax=axes[1],
                              alpha=0.7)
    ax.plot(sheet.face_df.loc[mesoderm_cells, 'y'],
            sheet.face_df.loc[mesoderm_cells, 'x'],
            'o', alpha=0.8, ms=5)

    fig.set_size_inches(24, 6)


def sagittal_view(sheet, min_slice, max_slice, face_mask=None,
                  coords=['x', 'y'], sagitta_axis='z', ax=None):
    """
    plot sagittal view
    """

    x, y = coords
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    sub_face_df = sheet.face_df[(sheet.face_df[sagitta_axis] > min_slice) & (
        sheet.face_df[sagitta_axis] < max_slice)]
    ax.plot(sub_face_df[x], sub_face_df[y], '.', color='black')

    if face_mask is None:
        return fig, ax

    sheet = sheet.extract(face_mask)
    sub_face_df = sheet.face_df[(sheet.face_df[sagitta_axis] > min_slice) & (
        sheet.face_df[sagitta_axis] < max_slice)]
    ax.plot(sub_face_df[x], sub_face_df[y], '.', color='red')

    return fig, ax


def color_info_view(sheet, face_information, color_map,
                    max_face_information=None, edge_mask=None,
                    edge_mask_color_map='hot', coords=['x', 'y'],
                    ax=None, normalization=1):
    """
    plot view sheet with :
    face color map according to face information
    and/or edge color map according to edge mask

    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    face_information : string
    color_map : string
    max_face_information : float
        max_value for the color map
    edge_mask : string
    edge_mask_color_map : string
    coords : list
        axis orientation of the plot

    """

    draw_specs = sheet_spec()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    perso_cmap = np.linspace(1.0, 1.0, num=sheet.face_df.shape[
        0]) * sheet.face_df[face_information]
    if max_face_information is None:
        max_face_information = max(perso_cmap)
    sheet.face_df['col'] = perso_cmap / max_face_information

    cmap_face = plt.cm.get_cmap(color_map)
    face_color_cmap = cmap_face(sheet.face_df.col)

    if edge_mask is not None:
        list_edge_in_mesoderm = sheet.edge_df['face'].isin(
            sheet.face_df[sheet.face_df[edge_mask]].index)

        cmap_edge = np.ones(sheet.edge_df.shape[
                            0]) * list_edge_in_mesoderm / normalization
        sheet.edge_df['col'] = cmap_edge / (max(cmap_edge)) / normalization

        if normalization == 1:
            cmap_edge = plt.cm.get_cmap(edge_mask_color_map)
        else:
            cmap_edge = plt.cm.get_cmap(edge_mask_color_map, normalization)

        edge_color_cmap = cmap_edge(sheet.edge_df.col)

        draw_specs['edge']['color'] = edge_color_cmap

    draw_specs['edge']['visible'] = True
    draw_specs['edge']['alpha'] = 0.7
    draw_specs['vert']['visible'] = False
    draw_specs['face']['visible'] = True
    draw_specs['face']['color'] = face_color_cmap
    draw_specs['face']['alpha'] = 0.6

    fig, ax = sheet_view(sheet, coords=coords, ax=ax, **draw_specs)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])

    return fig, ax


def three_axis_sheet_view(sheet, face_mask=None):
    """
    plot 3 view of the sheet :
    * ventral view
    * sagittal view
    * lateral view

    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    face_mask : string
    """

    if face_mask is not None:
        sheet_mesoderm = sheet.extract(face_mask)
    else:
        sheet_mesoderm = sheet

    edge_specs = {'alpha': 0.6,
                  'lw': 0.1,
                  'color': 'black'}
    scatter_specs = {'alpha': 0.8,
                     'ms': 5,
                     'color': 'red'}

    plt.figure(figsize=(18.5, 10.5))
    grid = gridspec.GridSpec(2, 2)
    axes_1 = plt.subplot(grid[0, 0])
    axes_2 = plt.subplot(grid[1, 0])
    axes_3 = plt.subplot(grid[:, 1])

    coord_pairs = [('z', 'y'), ('z', 'x'), ('x', 'y')]
    titles = ['lateral view', 'ventral view', 'sagittal view']
    axes = [axes_1, axes_2, axes_3]

    for coords, title, ax in zip(coord_pairs, titles, axes):
        u, v = coords
        fig, ax = quick_edge_draw(sheet,
                                  coords=coords,
                                  ax=ax,
                                  **edge_specs)

        ax.plot(sheet_mesoderm.face_df[u],
                sheet_mesoderm.face_df[v],
                'o', **scatter_specs)

        ax.set_title(title)
        ax.set_xlabel(u)
        ax.set_ylabel(v)
    return fig, axes


def save_ventral_plot(dirname, start=0, step=5, max_area=50,
                      face_caracteristic='area',
                      face_color='viridis', edge_color='bwr',
                      edge_normalization=1):

    dirname_ventral = os.path.join(dirname, 'ventral_view')
    try:
        os.mkdir(dirname_ventral)
    except FileExistsError:
        pass

    hfs = [f for f in os.listdir(dirname)
           if f.endswith('hf5')]

    for t in range(start, len(hfs), step):

        sheet = open_sheet(dirname, t)
        sub_sheet = sheet.extract_bounding_box(y_boundary=[0, 100])

        fig, ax = color_info_view(sub_sheet, face_caracteristic, face_color,
                                  max_area, 'is_mesoderm', edge_color,
                                  ['z', 'x'], normalization=edge_normalization)

        fig.set_size_inches(18.5, 10.5, forward=True)

        fig.savefig(os.path.join(dirname_ventral,
                                 'ventral_invagination_' + str(t)) + '.png')
        plt.close(fig)


def save_sagittal_plot(dirname, start=0, step=5, min_slice=-5, max_slice=5):

    dirname_sagital = os.path.join(dirname, 'sagittal_view')
    try:
        os.mkdir(dirname_sagital)
    except FileExistsError:
        pass

    hfs = [f for f in os.listdir(dirname)
           if f.endswith('hf5')]

    for t in range(start, len(hfs), step):

        sheet = open_sheet(dirname, t)

        fig, ax = sagittal_view(sheet, min_slice, max_slice,
                                face_mask='is_mesoderm',
                                coords=['x', 'y'], sagitta_axis='z')
        fig.set_size_inches(10, 10)
        plt.savefig(os.path.join(dirname_sagital,
                                 'invagination_' + str(t)) + '.png')
        plt.close(fig)


def save_3_axis_plot(dirname, start=0, step=1):
    """ Save 3 view plot of the sheet

    Create 3 view plot of the sheet :
        * sagittal view
        * ventral view
        * lateral view
    and save them in new directory called 'multiple_view'
    placed in the current directory.

    Parameters
    ----------
    directory : str
        complete directory path
    rev : bool
        start by the beginning or the end the simulation
    """
    dirname_multiple_view = os.path.join(dirname, 'multiple_view')
    try:
        os.mkdir(dirname_multiple_view)
    except FileExistsError:
        pass

    hfs = [f for f in os.listdir(dirname)
           if f.endswith('hf5')]

    for t in range(start, len(hfs), step):

        sheet = open_sheet(dirname, t)

        fig, [axes_1, axes_2, axes_3] = three_axis_sheet_view(
            sheet, 'is_mesoderm')

        fig.savefig(os.path.join(dirname_multiple_view,
                                 'invagination_' + str(t)) + '.png')
        plt.clf()
        plt.close("all")


def panel_sagittal_view(directory, df, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    t, d = define_time_max_depth(directory)
    df.loc[str(directory).split('/')[-1], 'time'] = t
    df.loc[str(directory).split('/')[-1], 'depth'] = d

    # Calcul du rapport de force
    sheet = open_sheet(directory, t)
    r = force_ratio(sheet)
    df.loc[str(directory).split('/')[-1], 'Fab/Fc'] = r
    print(directory)
    print(df.loc[str(directory).split('/')[-1]])

    # sagital view
    fig, ax = sagittal_view(
        sheet, -5, 5, 'is_mesoderm', ['x', 'y'], 'z', ax)
    a, c = 87, 87
    thetas = np.linspace(0, 2 * np.pi)
    ax.plot(c * np.cos(thetas), a * np.sin(thetas), color='grey', alpha=0.7)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(1, 2, round(r, 3), horizontalalignment='center',
            verticalalignment='center')
