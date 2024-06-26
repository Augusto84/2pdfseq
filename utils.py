import numpy as np
import math
from pylab import plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class UnionFind:
    parent = None
    n_elements = None
    n_elements_tree = None
    n_components = None
    ordem = None
    n_processed_elements = None

    def __init__(self, n_elements, old_uf = None):
        if old_uf is not None:
            self.n_elements = old_uf.n_elements
            self.parent = old_uf.parent
            self.n_elements_tree = old_uf.n_elements_tree
            self.n_components = old_uf.n_components
            self.ordem = old_uf.ordem
            self.n_processed_elements = old_uf.n_processed_elements
        else:
          self.n_elements = n_elements
          self.parent = np.arange(n_elements)
          self.n_elements_tree = np.ones(n_elements)
          self.n_components = n_elements
          self.ordem = np.zeros(n_elements)
          self.n_processed_elements = 0

    def _update_ordem(self, u):
        self.n_processed_elements += 1
        self.ordem[u] = self.n_processed_elements

    def get_ordem(self, u):
        return self.ordem[u]

    def _find (self, u):  # sub regions conformate with elements
        if u != self.parent[u]:
            self.parent[u] = self._find(self.parent[u])
        return self.parent[u]

    def _union (self, u, v):  #matriz de par ordenado
        p_u, p_v = map(self._find, [u, v]) #crea listas p_u, p_v que dependen de [u, v], definidos pela funcao  _find, p_u (pai de u)
        if self.n_elements_tree[p_v] > 1:  #se n_elements_tree na ubicacao p_v e maior que 1
            return False
        if p_u == p_v:                #se as listas p_u sao iguais p_v
            return False
        if self.n_elements_tree[p_u] < self.n_elements_tree[p_v]: # se o elemento p_u da lista n_elements_tree < ao elemento p_v da lista n_elements_tree
            self.parent[p_u] = p_v                           # o elemento p_u do parent = p_v
            self.n_elements_tree[p_v] += self.n_elements_tree[p_u]
        else:
            self.parent[p_v] = p_u                           # o p_u elemento da lista n_elements_tree = p_u
            self.n_elements_tree[p_u] += self.n_elements_tree[p_v]
        self.n_components -= 1                               # n_components = n_components -1
        return True

class Painter:
    EMPTY_CELL = -1#float('nan')
    # movimentos possiveis para uma determinada posicao x,em uma matriz 2D
    #  . 1 .
    #  2 x 3
    #  . 4 .
    # Conservando o ponto x, ele pode mover-se para os pontos 1, 2, 3 e 4
    # Isto e conhecido como ligada-4 uma vez que o ponto esta ligado ao seu 4-vizinhos

    move_row = [0, 0, 1, -1]
    move_col = [-1, 1, 0, 0]

    # To work with 8-connected, use the following varibles
    #move_row = [-1, -1, -1, 0, 0, 1, 1, 1]
    #move_col = [-1, 0, 1, -1, 1, -1, 0, 1]#[..., levels[i - 1][..., levels[i - 1][..., levels[i - 1][..., levels[i - 1]]]]]

    # Dada a dimensao da matriz e K, retorna uma matriz com K regioes.
    # as regiones poden ser iniciadas na lista index_element, entao K deve ser igual ao numero de elementos
    # cada regiao crece tanto quanto pudera
    # Nao se pode garantir que as regioes ira conter o mesmo numero de elementos
    def __init__(self, seeds, grid, uf, K):
        self.seeds = seeds
        self.index_next_move = np.arange(len(self.move_row))
        self.grid = grid
        self.K = K
        self.n_rows, self.n_cols, self.n_levels = grid.shape
        self.uf = uf
        map(self.uf._update_ordem, [x for x in self.seeds[:self.K]])
        self.regions = [[x] for x in self.seeds[:self.K]]

    def is_within_bounds(self, position, limits):               #define funcao que da os limites da matriz
        within = lambda a,x,b : a <= x < b
        #return [0] * len(position) <= position and position < limits
        return all(map(within, [0] * len(position), position, limits))

    def convert_number2coordinates(self, num, limits):
        n_rows, n_cols, n_levels = limits
        depth = num / (n_rows * n_cols)
        rem   = num % (n_rows * n_cols)
        row, col = rem / n_cols, rem % n_cols
        return (row, col, depth)

    def convert_coordinates2number(self, coordinates, limits):
        # coordinates = row, col, depth
        # limits = n_rows, n_cols, levels
        row, col, depth = coordinates
        n_rows, n_cols, n_levels = limits
        position =  depth * n_rows * n_cols + row * n_cols + col
        return position

    def test_converter(self):
        limits = (4,5,1)
        n = 4 * 5 * 1
        A = np.arange(n).reshape(limits)
        print A.reshape(4,5)
        print 'i : r c d A n'
        for i in xrange(n):
            r,c,d = self.convert_number2coordinates(i, limits)
            print i, ':', r, c, d, A[r,c,d], self.convert_coordinates2number([r,c,d], limits)
            assert(A[r, c, d] == i)
            assert(self.convert_coordinates2number([r,c,d], limits) == i)
        return True

    def paint_Kregions (self, K, iter_label, max_steps = 250):                     #define a funcao paint_Kregions#colocar iter
        np.random.shuffle(self.index_next_move)
        np.random.shuffle(self.seeds)
        ith_step = 0
        all_regions_found = False
        stop = False
        cannot_expand_more = 0 #float('nan') #modificado
        limits = [self.n_rows, self.n_cols, self.n_levels]
        while not all_regions_found and ith_step < max_steps and cannot_expand_more < K:
            cannot_expand_more = 0 #float('nan') #modificado
            for ith_region in xrange(K):                    #para as K regioes
                if self.uf.n_components == K:                #se a funcao _get_n_componets() == K
                    all_regions_found = True                #tudas as regiones all_regions_found = True
                    break
                found_valid_move = False
                n_elements_current_region = len(self.regions[ith_region]) #conta o numero de elementos das regioes
                #stop = not (stop | (n_elements_current_region > 0))
                #
                while not found_valid_move and n_elements_current_region > 0: #while (not False) and (n_elements_current_region > 0)
                    u = self.regions[ith_region][0]
                    row, col, depth = self.convert_number2coordinates(u, limits)
                    #
                    for i_move in self.index_next_move:
                        nxt_row, nxt_col = row + self.move_row[i_move], col + self.move_col[i_move]
                        if not self.is_within_bounds((nxt_row, nxt_col), limits[:2]) \
                            or math.isnan(self.grid[(nxt_row, nxt_col, depth)]): #float('nan'): #change 0
                            continue
                        v = self.convert_coordinates2number((nxt_row, nxt_col, depth), limits)
                        #v = sao os elementos aderidos por a sequencia en cruz
                        if self.uf._union(u, v):
                            found_valid_move = True
                            self.uf._update_ordem(v)
                            break
                    if not found_valid_move:
                        self.regions[ith_region] = self.regions[ith_region][1:]
                        n_elements_current_region -= 1
                    else:
                        ith_step += 1
                        self.regions[ith_region].append(v)
                    np.random.shuffle(self.index_next_move)
                np.random.shuffle(self.regions[ith_region])
                if math.isnan(n_elements_current_region): #(n_elements_current_region)#float('nan'): #o (reemplazed)
                    cannot_expand_more += 1
        #
        if cannot_expand_more == K:
            print 'it was not possible to expand more due to connectivity issues'
        #
        #self.connected_map = np.array([self.uf._find(i) for i in xrange(self.uf.n_elements)]).reshape(limits)
        #self.connected_map_ordem = np.array([self.uf.get_ordem(i) for i in xrange(self.uf.n_elements)]).reshape(limits)
        #self.connected_processed_map = self.connected_map_ordem > 0
        self.connected_map = np.empty(limits, int)
        self.connected_map_ordem = np.empty(limits, int)
        self.connected_processed_map = np.empty(limits, int)
        for i in xrange(self.uf.n_elements):
            coord = self.convert_number2coordinates(i, limits)
            self.connected_map[coord] = self.uf._find(i)
            self.connected_map_ordem[coord] = self.uf.get_ordem(i)
            self.connected_processed_map[coord] = self.uf.get_ordem(i) > 0

    def test2D(self, showPlots = False):
        assert(self.test_converter())
        import hashlib
        # to test run the following command on a terminal:
        # python -c "from utils import *; guto = Painter([0], np.zeros([1,1,1]), UnionFind(1), 1); guto.test2D(showPlots=True);"
        # Creating varibles to set up the Painter
        np.random.seed(1)
        n_rows, n_cols, n_levels = 20, 20, 1
        grid = np.random.randint(low = 0, high = 100, size = n_rows * n_cols * n_levels).reshape([n_rows, n_cols, n_levels])
        grid = grid * (grid > np.random.randint(low = 0, high = 30))
        K = 3
        seeds = np.random.randint(low = 0, high = n_rows * n_cols * n_levels, size = K)
        assert(all([grid[self.convert_number2coordinates(ith_seed, [n_rows, n_cols, n_levels])] > 0 for ith_seed in seeds]))
        uf = UnionFind(n_rows * n_cols * n_levels)
        self.__init__(seeds, grid, uf, K)
        #
        def showGrid(grid_):
            if not showPlots:
                return
            plt.imshow(grid_[..., 0], interpolation='nearest')
            plt.show()
        # Show grid
        showGrid(grid)
        # Run the painter
        self.paint_Kregions(K, iter_label=None, max_steps=100)
        # Show results
        showGrid(self.connected_map)
        hash_obj = hashlib.md5(self.connected_map.reshape(-1))
        assert(hash_obj.hexdigest() == "d6b6444b088f017e4a500a10a80761a8")


class Map3D:
    def __init__(self, map_name = ''):
        self.map_name = map_name

    def setVectors(self, X, Y, Z, V):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.V = V

    def readFromFile(self, filename_path):
        reader = open(filename_path, 'r')
        self.map_name = reader.readline()
        reader.readline()
        reader.readline()
        reader.readline()
        reader.readline()
        reader.readline()
        self.X, self.Y, self.Z, self.V = [], [], [], []
        for line in reader:
            x, y, z, v = map(float, [t for t in line.replace('\t', ' ').split(' ') if len(t) > 0])
            self.X.append(int(x))
            self.Y.append(int(y))
            self.Z.append(int(z))
            self.V.append(v)
        reader.close()

    def convertCoordinates(self):
        realCoordinates = lambda fake_coordinates: dict([(val, idx) for idx, val in enumerate(sorted(set(fake_coordinates)))])
        mapCoordinates = lambda fake_coordinates, real_coordinates: map(real_coordinates.get, fake_coordinates)

        self.realX, self.realY, self.realZ = map(realCoordinates, [self.X, self.Y, self.Z])
        self.nX, self.nY, self.nZ = map(mapCoordinates, [self.X, self.Y, self.Z], [self.realX, self.realY, self.realZ])
        self.n_cols, self.n_rows, self.n_depth = map(len, [self.realX, self.realY, self.realZ])

    def toMatrix(self):
        M = np.empty([self.n_rows, self.n_cols, self.n_depth], np.float)
        for idx in xrange(len(self.V)):
            M[self.nY[idx], self.nX[idx], self.nZ[idx]] = self.V[idx]
        return M

    def toList(self, M):
        L = [0] * len(self.V)
        for idx in xrange(len(self.V)):
            L[idx] = M[self.nY[idx], self.nX[idx], self.nZ[idx]]
        return L

    def plotSurface(self, level, M):
        n_elts_per_levels = self.n_rows * self.n_cols
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        xx, yy = np.meshgrid(range(self.n_cols), range(self.n_rows))
        surf = ax.plot_surface(xx, yy,  M[..., level], cmap = 'terrain', cstride = 1, rstride = 1, linewidth=0, antialiased=False)

        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

#from utils import Map3D;
#mapa = Map3D('./3dproob3.txt');
#mapa.plotSurface(0, matrix)
