import numpy as np
import pandas as pd

!pip install lmfit -q
from lmfit import Model, Parameters, minimize, report_fit

from pandas.api.types import is_list_like   # list, tuple, series, NOT string
from pandas.api.types import is_dict_like



class CRNDataModel(Model) :
  """
  lmfit.Model plus:
  > Time
    - independent_vars = 't'
    - self.t_eval stores ordered list-like of times that need to be evaluated for fitting, plotting, etc.
  > Parameters
    - self.params (type lmfit.Parameters) for storing fitted vals
      - set in __init__
      - automatically nonnegative
      - lmfit.Model self.make_params() is not stored instance var
    - self.__init__ argument 'param_names' can either be dict-like or list-like
      e.g. param_names = {'j':2,'k':3} equivalent to param_names = ('j0','j1','k0','k1','k2')
  """

  def __init__(self, func, t_eval, emp_data, param_names=None, name=None) :
    # necessary imports here

    
    
    if(is_dict_like(param_names)) :
      param_names = (f"{var}{n}" for n in range(num) for var,cnt in param_names.items())
    else :
      assert is_list_like(param_names), f'{type(self)}.__init__ parameter \'param_names\' should be list-like or dict-like. (e.g. param_names = {{\'j\':2,\'k\':3}} equivalent to param_names = (\'j0\',\'j1\',\'k0\',\'k1\',\'k2\') )'

    super().__init__(func, param_names=param_names, name=name)

    self.t_eval = t_eval
    self.emp_data = emp_data

    assert is_list_like(t_eval), f'{type(self)}.__init__ parameter \'t_eval\' should be list-like of times to evaluate model at'
    self.t_eval = t.eval

  def fit(self) :
    return super().fit(self.emp_data)

  
    # result = gmodel.fit(data)



class Kurchak1Model(CRNDataModel) :
  
  def __init__(self, t_eval, emp_data, zerok4k5:int, name=None) :

    assert 0<=zerok4k5 and zerok4k5<3, f'{type(self)}.__init__ argument \'zerok4k5\' must be between 0 and 2. (e.g. 0 = both non-zero, 1 = only k5 zero, 2 = only k4 zero)'
    assert is_list_like(t_eval) , f'{type(self)}.__init__ argument \'t_eval\' must be list-like.'
    assert is_list_like(emp_data) , f'{type(self)}.__init__ argument \'emp_data\' must be list-like.'

    super(self.approx_fnct, t_eval, emp_data, name=name,
          param_names=['k0','k1','k2','k3','k4','k5','fl_mult','D_init'])

    for i in range(4) :
      self.set_param_hint(param_names[i],value=1,min=0)
    self.set_param_hint(param_names[4],value=0,vary=zerok4k5<2,min=0)
    self.set_param_hint(param_names[5],value=0,vary=zerok4k5!=1,min=0)
    self.set_param_hint(param_names[6],value=1,min=1)
    self.set_param_hint(param_names[7],value=400,min=0)

  # to be passed to parent class Model
  # accepts all IVs for Kurchak1 system
    # only t_eval independent (NOT TO BE FITTED)
  # returns array of y-vals same size as t_eval
  def approx_fnct (self,k0,k1,k2,k3,k4,k5,fl_mult,D_init) :
    t_span = (t_eval[0],t_eval[len(t_eval)-1])
    def fl_ODE(D,G) :
      return k3*(D_init-G) -(k0+k3)*D +fl_mult*( (D_init-D)*(k1+k2*G) -G*(k1+k4) -G**2*(k2+k5) )
    return solve_ivp(fl_ODE, t_span, (D_init,0), t_eval=t_eval)
    # returns array of fluorescence values
    # default uses RK45
