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
