#!/bin/python
import sgems
import math
import numpy as np
import scipy



"""
The base python sgems plugin.
"""
def read_params(a,j=''):
  for i in a:
    if (type(a[i])!=type({'a':1})):
      print j+"['"+str(i)+"']="+str(a[i])
    else:
      read_params(a[i],j+"['"+str(i)+"']")

class local:
    def __init__(self):
        pass

    def initialize(self, params):
        self.params=params
        return True

    def execute(self):
        read_params(self.params)
        #print self.params
        #print (self.params)

        head_property = self.params['head_prop']['property']
        head_grid = self.params['head_prop']['grid']
        g = None
        ind = None
        w = None
        m = None

        x = float(self.params['x']['value'])
        y = float(self.params['y']['value'])
        z = float(self.params['z']['value'])

        xgrid = sgems.get_property(head_grid,"_X_")
        ygrid = sgems.get_property(head_grid,"_Y_")
        zgrid = sgems.get_property(head_grid,"_Z_")
        n = sgems.get_property(head_grid,head_property)
        r = sgems.get_dims(head_grid)

        def id_search(g,w,m):
            ind = sgems.get_closest_nodeid(head_grid,x,y,z)
            print ind
            #g = sgems.get_ijk(head_grid,ind)
            #w = sgems.get_ijk(head_grid,(len(l_c)-1))
            #m = sgems.get_ijk(head_grid,(min(l_c)))

            #def ijk_in_n(head_grid, i, j, k):
            #dims = sgems.get_dims(head_grid)
            #n_ = g[2]*dims[0]*dims[1]+g[1]*dims[0]+g[0]
            #print " ID range [",m," to ",w,"]", " ID select is ",n_ #ijk_in_n('GRID2', 24, 29, 0)
                    #print "start id ",g, " id max ",w," id min ",m
        id_search(g,w,m)

        #def id_search(g,w,m):
        #    ind = sgems.get_closest_nodeid(head_grid,x,y,z)
            #print " ID of point is" ind
        #print id_search(x,y,z)

        return True

    def finalize(self):
        return True

    def name(self):
        return "local"

################################################################################
def get_plugins():
    return ["local"]

################################################################################
