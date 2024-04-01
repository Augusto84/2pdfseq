#!/bin/python
import sgems
import copy
import math
import numpy as np
from utils import UnionFind, Painter, Map3D
from pylab import plt


"""
The base python sgems plugin.
"""
class pdfsequential12:
    def __init__(self):
        pass

    def initialize(self, params):
        self.params=params
        return True

    def execute(self):
        # Map properties
        head_property = self.params['head_prop']['property']
        head_grid = self.params['head_prop']['grid']

        # Maps to generate as output
        new_prop = self.params['new_prop']['value']
        map_seed_prefix_name = self.params['new_prop1']['value']
        map_seq_prefix_name  = self.params['new_prop2']['value']

        # Cutoff paramters
        cut_off = float(self.params['cut_off']['value'])
        less_equal = self.params['rsless']['value']
        more_equal = self.params['rbgreater']['value']

        # Parameters
        fases = int(self.params['fases']['value'])
        simulation = int(self.params['simulation']['value'])
        runs = int(self.params['runs']['value'])

        # Seeds
        seeds = self.params['seeds']['value']
        seeds_ = map(int,seeds.strip().split(","))

        # @TODO: This variable is probably be used in the future
        # datahist_global = None

        # Coordinates for x,y,z for each value in grid_value
        xgrid = sgems.get_property(head_grid,"_X_")
        ygrid = sgems.get_property(head_grid,"_Y_")
        zgrid = sgems.get_property(head_grid,"_Z_")
        grid_value = sgems.get_property(head_grid, head_property)
        r = sgems.get_dims(head_grid)

        result_cut_off = None #list cut off
        h = 1

        n_rows = r[0]#30#len(ygrid)
        n_cols = r[1]#25#len(xgrid)
        n_levels = r[2]
        K = len(seeds_)
        print "Map info dimensions: ", r, n_rows, n_cols, n_levels

        def filter_cell_value_by_cut_off(pass_cut_off):
            result_cut_off = [cell_value if pass_cut_off(cell_value) else 0 for cell_value in grid_value]
            return result_cut_off

        pass_cut_off = None
        if more_equal == '1' and less_equal == '0':
            pass_cut_off = lambda value, threshold = cut_off: value >= threshold
        elif more_equal == '0' and less_equal == '1':
            pass_cut_off = lambda value, threshold = cut_off: value <= threshold
        elif more_equal == '0' and less_equal == '0':
            print "Select option of cut off"
            self.dict_gen_params['execution_status'] = "ERROR"
        else:
            print "Error in execution type parameters"
            self.dict_gen_params['execution_status'] = "ERROR"
        if pass_cut_off is None:
            return False

        # Run cut off
        result_cut_off = filter_cell_value_by_cut_off(pass_cut_off)
        result_cut_off = [float('nan') if x == 0 else x for x in result_cut_off]
        # Update property in sgems
        sgems.set_property(head_grid, new_prop, result_cut_off)
        result_cut_off_clean_zero = filter(lambda x:x>0,result_cut_off)
        #sgems.set_property(head_grid, "new_prop", result_cut_off_clean_zero)#show number values without order
        # Transform the list values into a 3D-matrix
        mapa = Map3D()
        mapa.setVectors(xgrid, ygrid, zgrid, result_cut_off)
        mapa.convertCoordinates()
        grid = mapa.toMatrix()

        #plt_imshowN([grid, grid == 0, grid, grid == 0],
        #Max        #            [0, 0, 1, 1], [dict(cmap='jet', interpolation='nearest')] * 4, dim=(2,2))

        def quantil(p):
            a,b = np.histogram(p, bins=20)
            return a#newversion
            #return np.percentile(p, np.arange(0, 100, 5))

        def compare_histograms(s, h):
            return sum([(i - j) ** 2 / (i + j) for i in s for j in h if i + j > 0])

        c_global = quantil(result_cut_off_clean_zero)

        def run_union_find(painter, K, iter_label, max_steps):
            fake_painter = copy.deepcopy(painter)
            fake_painter.paint_Kregions(K, iter_label, max_steps)
            y0 = mapa.toList(fake_painter.connected_processed_map * fake_painter.grid)
            y0_clean_zero = filter(lambda x:x>0,y0)
            c_list = quantil(y0_clean_zero)

            return compare_histograms(c_global, c_list), fake_painter

        def search_best_run(painter, K, n_iterations, max_steps):
            best_cost = 10000000000# @TODO: rename variable as: max_possible_cost
            best_painter = None
            for iter_label in xrange(n_iterations):
                cost, fake_painter = run_union_find(painter, K, iter_label, max_steps)
                print 'iter', iter_label, 'chb', cost
                if cost < best_cost:
                    best_cost = cost
                    best_painter = fake_painter
            assert(best_painter != None)
            return (best_cost, best_painter)

        def repeat_all(grid,K, painter, n_iterations, n_steps_list, n_runs):
            for i, max_steps in zip(xrange(n_runs), n_steps_list):
                print i, "run"
                cost, painter = search_best_run(painter, K, n_iterations, max_steps)
                #
                maps_seed = (painter.connected_map * painter.connected_map)
                maps_seed_ = mapa.toList(maps_seed)
                #sgems.set_property(head_grid, map_seed_prefix_name + '-' + str(i), maps_seed_)
                #
                maps_ = painter.connected_map_ordem
                print (maps_.shape)
                maps__ = mapa.toList(maps_)
                #sgems.set_property(head_grid, "seq_map"+'-'+str(i),maps__)#adicionado teste
                #sgems.set_property(head_grid, map_seq_prefix_name +'-'+str(i),maps__)
                maps___ = [float('nan') if x == 0 else x for x in maps__]
                sgems.set_property(head_grid, map_seq_prefix_name + '-' + str(i), maps___)
                #
                ee = np.ma.masked_array(grid,mask=maps_==0)
                ee_ = ee.filled(0)
                ee__ = mapa.toList(ee_)
                ee__ = [float('nan') if x == 0 else x for x in ee__]
                #sgems.set_property(head_grid, "sect_maps"+'-'+str(i),ee__)
                sgems.set_property(head_grid, map_seed_prefix_name+'-'+str(i),ee__)

        uf = UnionFind(n_rows * n_cols * n_levels)
        guto = Painter(seeds_, grid, uf, K)
        n_steps_list = [fases] * runs # [fases, fases, fases, .... runs_veces, ..., fases]
        n_steps_list[0] -= len(seeds_)# [fases-len(seeds), fases, fases, .... runs_veces, ..., fases]
        repeat_all(grid, K, painter = guto, n_iterations = simulation, n_steps_list = n_steps_list, n_runs = runs)#cambiando

        return True

    def finalize(self):
        return True

    def name(self):
        return "pdfsequential12"

################################################################################
def get_plugins():
    return ["pdfsequential12"]

################################################################################
from pylab import plt
import numpy as np

def plt_imshowN(imgs, levels, params = None, dim = (1,2)):
    fig = plt.figure()
    for i in xrange(1, np.multiply(*dim) + 1):
        p = fig.add_subplot(dim[0], dim[1], i)
        if params is not None and params[i-1] is not None:
            p.imshow(imgs[i-1][..., levels[i - 1]], **params[i-1])
        else:
            p.imshow(imgs[i-1][..., levels[i - 1]])
    plt.show()

def plotSurface(M, level):
    n_elts_per_levels = self.n_rows * self.n_cols
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    xx, yy = np.meshgrid(range(self.n_cols), range(self.n_rows))
    surf = ax.plot_surface(xx, yy,  M[..., level], cmap = 'terrain', cstride = 1, rstride = 1, linewidth=0, antialiased=False)
    plt.show()

# plt_imshow([im1, im2], [None, {'cmap':'gray'}], dim=(1,2))"""
