import sgems
import math
import numpy as np


def read_params(a,j=''):
  for i in a:
    if (type(a[i])!=type({'a':1})):
      print j+"['"+str(i)+"']="+str(a[i])
    else:
      read_params(a[i],j+"['"+str(i)+"']")

class topo:
    def __init__(self):
        pass

    def initialize(self, params):
        self.params=params
        return True

    def execute(self):
        head_property = self.params['head_prop']['property']
        head_grid = self.params['head_prop']['grid']
        new_prop = self.params['new_prop']['value']


        zgrid=sgems.get_property(head_grid,'_Z_')
        xgrid=sgems.get_property(head_grid,'_X_')
        ygrid=sgems.get_property(head_grid,'_Y_')
        grid_value=sgems.get_property(head_grid,head_property)
        #d=sgems.get_ijk('grid3',254)
        #print d
        #dimentions
        r=sgems.get_dims(head_grid)

        n_rows = r[0]
        n_cols = r[1]
        n_levels = r[2]

        def ip(grid, i, j, k):
            dims = sgems.get_dims(grid)
            ip = k*dims[0]*dims[1]+j*dims[0]+i
            return ip

        #print type(ip('grid3',1,2,3))

        for i in range(0, n_rows):
        	for j in range(0, n_cols):
        		for k in range(0, n_levels):
        #			print grid_value[ip('grid3',i,j,k)]
        			if np.isnan(grid_value[ip(head_grid, i, j, k)]):
        				continue
        			is_last = True
        			for m in range(k+1, n_levels):
        				if not np.isnan(grid_value[ip(head_grid, i, j, m)]):
        					is_last = False
        					break
        			if not is_last:
        				grid_value[ip(head_grid, i, j, k)] = np.nan
        #print grid_value
        sgems.set_property(head_grid,new_prop,grid_value)

        return True

    def finalize(self):
        return True

    def name(self):
        return "topo"

################################################################################
def get_plugins():
    return ["topo"]

################################################################################
