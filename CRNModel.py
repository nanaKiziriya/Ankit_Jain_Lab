import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!pip install lmfit -q
from lmfit import Model, Parameters, minimize, report_fit
from scipy.integrate import solve_ivp

import inspect

from pandas.api.types import is_list_like   # list, tuple, series, NOT string
from pandas.api.types import is_dict_like

from abc import ABC, abstractmethod

################################################################

#@title Abstract class: CRN Model
#@markdown parent class for all other CRN models in this program

class CRNModel(Model,ABC) :

  model_cnt=0

  '''
  lmfit.Model plus:
  > Time
    - independent_vars = 't_eval'
    - self.t_eval stores ordered list-like of times that need to be evaluated for fitting, plotting, etc.
  > Parameters
    - self.params (type lmfit.Parameters) for storing fitted vals
      - set in __init__
      - automatically nonnegative
      - lmfit.Model self.make_params() is not stored instance var
    - self.__init__ argument 'param_names' can either be dict-like or list-like
      e.g. param_names = {'j':2,'k':3} equivalent to param_names = ('j0','j1','k0','k1','k2')
  '''


  def __init__(self,
               t_eval, emp_data, emp_params:Parameters,   # self
               name=None) :              # super

    # argument validation
    assert is_list_like(t_eval), f'\'t_eval\' should be list-like: time measurements \nInvalid input: {t_eval}'
    assert is_list_like(emp_data), f'\'emp_data\' should be list-like: data measurements over time \nInvalid input: {emp_data}'

    temp_pnames = inspect.signature(self.func).parameters
    # if 'self' in temp_pnames : temp_pnames.pop(0)
    # assert 'self' not in temp_pnames, f'\'self\' failed to be removed from \'param_names\''

    assert set(emp_params.valuesdict().keys()).issubset(temp_pnames), f'\'emp_params\':Parameters argument should only define \'func\':callable parameters \nInvalid input: {set(emp_params.valuesdict().keys()).difference(temp_pnames)}'

    self.t_eval = t_eval
    self.emp_data = emp_data
    self.emp_params = emp_params

    if not name :
      name = f'CRNModel_{model_cnt}'
      model_cnt+=1

    # super constructor
    super().__init__(self.func,
                     independent_vars=['t'],
                     name=name)

  @abstractmethod
  # subclass must only implement parameters to be fit (excl. 'self', time 't')
  def func(self,t,*args,**kwargs):
    pass

  def fit(self) :
    return super().fit(self.emp_data,params=self.emp_params,t=self.t_eval)

################################################################

#@title Finke-Watzky Model
#@markdown shown to not be sufficient for Boateng1 system (June 22, 2026)

# Logistic CRN model!
class FinkeWatzkyModel(CRNModel) :

  model_cnt = 0

  def __init__(self,
               t_eval, emp_data,  # super
               name=None):        # super

    # argument validation
    assert is_list_like(t_eval) , f'{type(self)}.__init__ argument \'t_eval\' must be list-like.'
    assert is_list_like(emp_data) , f'{type(self)}.__init__ argument \'emp_data\' must be list-like.'

    if not name :
      name = f'FinkeWatzkyModel_{model_cnt}'
      model_cnt+=1

    # Set empirical limits on Parameters
    # (name, value=None, vary=True, min=-inf, max=inf, expr=None, brute_step=None)
    emp_params = Parameters()
    emp_params.add('k0', value=10, min=0) # nucleation rate
    emp_params.add('k1', value=10, min=0) # aggregation rate
    emp_params.add('fl_mult', value=10, min=0) # fl_G
    emp_params.add('M_init', value=10, min=0.1) # intial species

    # super constructor
    super().__init__(t_eval = t_eval,
                     emp_data = emp_data,
                     emp_params = emp_params,
                     name = name)

  def func(self,t, k0,k1,fl_mult,M_init): # EXPLICIT SOLUTION
    s = k0+k1*M_init
    M = s*M_init/(M_init*k1+k0*np.exp(s)*t)
    G = M_init-M
    return fl_mult*G # degree of aggr. ~ fluorescence
    # fl_M nontrivial due to lack of other fl-contributing species
