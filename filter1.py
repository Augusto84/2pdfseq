#!/bin/python
import sgems
import numpy as np
import math


"""
The base python sgems plugin.
"""
def read_params(a,j=''):
  for i in a:
    if (type(a[i])!=type({'a':1})):
      print j+"['"+str(i)+"']="+str(a[i])
    else:
      read_params(a[i],j+"['"+str(i)+"']")

class filter1:
    def __init__(self):
        pass

    def initialize(self, params):
        self.params=params
        return True

    def execute(self):

        head_property = self.params['head_prop']['property']
        head_grid = self.params['head_prop']['grid']

        tail_property = self.params['tail_prop']['property']
        tail_grid = self.params['tail_prop']['grid']

        cut_off_first = float(self.params['cut_off_first']['value'])
        new_prop = self.params['new_prop']['value']

        grid_value1 = sgems.get_property(head_grid, head_property)
        grid_value2 = sgems.get_property(tail_grid, tail_property)

        more_equal = self.params['greater']['value']
        less_equal = self.params['less']['value']

        def cut_off_w(grid_value, cut_off,more_equal,less_equal):
            result=range(len(grid_value))#more_equal, less_equal = map(int, [more_equal, less_equal])
            for i in xrange(0, len(grid_value)):
                if (cut_off <= grid_value[i] or np.isnan(grid_value[i])) and (more_equal =='a' or less_equal =='0') :
                    result[i]=grid_value[i]
                elif (cut_off >= grid_value[i] or np.isnan(grid_value[i])) and (more_equal =='0' or less_equal =='a'):
                    result[i]=grid_value[i]
                else:
                    result[i] = np.nan
            return result


        A1s = cut_off_w(grid_value1,cut_off_first,more_equal,less_equal)
        #sgems.set_properties(head_grid, 'resul_pre', A1s)
        #Define function fitro
        def filtro(grid_value_2,grid_final):
            result2=range(len(grid_value_2))
            for i in xrange(0,len(grid_value_2)):
                if np.isnan(grid_value_2[i]):
                    result2[i]=grid_value_2[i]
                else:
                    result2[i]=grid_final[i]
            return result2

        #print filtro(A1,tail_property)

        k=filtro(A1s,grid_value2)
        print k
        print len(k)
        #        k_=toList(k)

        sgems.set_property(head_grid,new_prop,k)
        return True

    def finalize(self):
        return True

    def name(self):
        return "filter1"

################################################################################
def get_plugins():
    return ["filter1"]

################################################################################
