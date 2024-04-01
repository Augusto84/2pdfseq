import sgems
import math
import numpy as np

"""
The base python sgems plugin.
"""
def read_params(a,j=''):
  for i in a:
    if (type(a[i])!=type({'a':1})):
      print j+"['"+str(i)+"']="+str(a[i])
    else:
      read_params(a[i],j+"['"+str(i)+"']")

class pushback:
    def __init__(self):
        pass

    def initialize(self, params):
        self.params=params
        return True

    def execute(self):
        head_property = self.params['head_prop']['property']
        head_grid = self.params['head_prop']['grid']


        head_property1 = self.params['head_prop1']['property']
        head_grid1 = self.params['head_prop1']['grid']

        new_prop = self.params['new_prop']['value']



        zgrid=sgems.get_property(head_grid,'_Z_')
        xgrid=sgems.get_property(head_grid,'_X_')
        ygrid=sgems.get_property(head_grid,'_Y_')
        grid_value1=sgems.get_property(head_grid,head_property)
        grid_value2=sgems.get_property(head_grid,head_property1)


        #B=[1,2,3,4,np.nan,2,np.nan,np.nan,np.nan,np.nan,np.nan]
        #A=[1,np.nan,3,np.nan,np.nan,2,np.nan,np.nan,np.nan,np.nan,np.nan]
        def restnan(list1,list2):
            result=range(len(list1))
            for i in xrange(len(list1)):
                if np.isnan(list2[i]) or np.isnan(list1[i]):
                    result[i]=list2[i]
                else:
                    if list2[i] == list1[i]:
                        result[i] = np.nan
            return result

        cut2 = restnan(grid_value1,grid_value2)



        sgems.set_property(head_grid,new_prop,cut2)

        return True

    def finalize(self):
        return True

    def name(self):
        return "pushback"

###############################################################################
def get_plugins():
    return ["pushback"]

################################################################################
