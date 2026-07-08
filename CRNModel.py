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

################################################################################################################################

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

################################################################################################################################

#@title Kurchak1 Model v.1, not maximal (uses fl_mult)

class Kurchak1Model(CRNModel) :

  model_cnt = 0
  method = 'RK45'

  def __init__(self,
               t_eval, emp_data,  # super
               name=None,         # super
               method='RK45',     # self
               zerok4k5:int=0) :  # self

    self.method = method

    # argument validation
    assert 0<=zerok4k5 and zerok4k5<3, f'{type(self)}.__init__ argument \'zerok4k5\' must be between 0 and 2. \n(e.g. 0 = both non-zero, 1 = only k5 zero, 2 = only k4 zero)'
    assert is_list_like(t_eval) , f'{type(self)}.__init__ argument \'t_eval\' must be list-like.'
    assert is_list_like(emp_data) , f'{type(self)}.__init__ argument \'emp_data\' must be list-like.'

    if not name :
      name = f'Kurchak1Model_{model_cnt}'
      model_cnt+=1

    # Set empirical limits on Parameters
    # (name, value=None, vary=True, min=-inf, max=inf, expr=None, brute_step=None)
    emp_params = Parameters()
    emp_params.add('k0', value=.4, min=0) # decomposition
    emp_params.add('k1', value=.2, min=0) # nucleation
    emp_params.add('k2', value=0, min=0) # aggregation
    emp_params.add('k3', value=.03, min=0) # rev. decomp
    emp_params.add('k4', value=0, vary=zerok4k5!=2, min=0) # rev. nucl
    emp_params.add('k5', value=0, vary=zerok4k5!=1, min=0) # rev. aggr
    emp_params.add('fl_mult', value=10, min=1) # fl_G/fl_D
    # emp_params.add('fl_mult', value=10, vary=False, min=1) # fl_G/fl_D
    emp_params.add('D_init', value=1, min=0.1) # intial species

    # super constructor
    super().__init__(t_eval = t_eval,
                     emp_data = emp_data,
                     emp_params = emp_params,
                     name = name)

  def func(self,t, k0,k1,k2,k3,k4,k5,fl_mult,D_init):
    t = np.asarray(t)
    t_span = (t[0], t[-1])

    def rhs(T, y):
      D,G = y
      dDdt = k3*(D_init - G) - (k0 + k3)*D
      dGdt = (D_init - D)*(k1 + k2*G) - G*(k1 + k4) - G**2*(k2 + k5)
      return [dDdt, dGdt]

    # def inner_func(x,Aggr:np.ndarray) :
    #   nonlocal t, k0,k1,k2,k3,k4,k5,fl_mult,D_init
    #   D = Aggr[0]
    #   G = Aggr[1]
    #   return np.array(k3*(D_init-G) - (k0+k3)*D, # dD/dt
    #           (D_init-D)*(k1+k2*G) - G*(k1+k4) - G**2*(k2+k5)) # dG/dt

    # default 'RK45', alt 'BDF' 'RK23' 'DOP853' 'Radau' 'LSODA'
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#:~:text=methodstr
    sol = solve_ivp(rhs, t_span, y0 = [D_init,0],
        dense_output=True,atol=1e-6, rtol=1e-6,
        # t_eval = t,
        method=self.method
        # BDF seems to work best so far, though they all kinda suck for this
        # Maybe It's better to normalize AFTER?
        # A: YES, scale down* after fitting, else truncation error -> stiff
        #
        )

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    Y = sol.sol(t)    # shape (n_states, len(t_obs))
    # Y = sol.y   # shape (2, len(t))
    return Y[0] + fl_mult * Y[1]

################################################################################################################################

#@title Boateng1 Model v.1_full
#@markdown stiff

class Boateng1Model(CRNModel) :

  model_cnt = 0
  method = 'RK45'

  def __init__(self,
               t_eval, emp_data,  # super
               name=None,         # super
               method='RK45') :   # self

    self.method = method

    # argument validation
    assert is_list_like(t_eval) , f'{type(self)}.__init__ argument \'t_eval\' must be list-like.'
    assert is_list_like(emp_data) , f'{type(self)}.__init__ argument \'emp_data\' must be list-like.'

    if not name :
      name = f'Boateng1Model_{model_cnt}'
      model_cnt+=1

    # Set empirical limits on Parameters
    # (name, value=None, vary=True, min=-inf, max=inf, expr=None, brute_step=None)
    emp_params = Parameters()
    emp_params.add('k0', value=1, vary=True, min=0) # 2nd aggregation
    emp_params.add('k1', value=1, min=0) # nucleation
    emp_params.add('k2', value=1, min=0) # 1st aggregation
    emp_params.add('k3', value=1, vary=True, min=0) # rev. 2nd aggr
    emp_params.add('k4', value=0, vary=True, min=0) # rev. nucl
    emp_params.add('k5', value=0, vary=True, min=0) # rev. aggr
    emp_params.add('fl_A', value=1.0, min=0.01) # fl_A
    emp_params.add('fl_B', value=1.0, min=0.01) # fl_B
    emp_params.add('fl_C', value=1.0, min=0.01) # fl_C
    emp_params.add('A_init', value=1, min=0.01) # initial conc. A
    emp_params.add('B_init', value=0.1, min=0.01) # initial conc. B
    emp_params.add('C_init', value=0.1, vary=True, min=0.01) # initial conc. C

    # super constructor
    super().__init__(t_eval = t_eval,
                     emp_data = emp_data,
                     emp_params = emp_params,
                     name = name)

  def func(self,t, k0,k1,k2,k3,k4,k5, fl_A,fl_B,fl_C, A_init,B_init,C_init):
    t = np.asarray(t)
    t_span = (t[0], t[-1])

    def rhs(T, y):
      # D,G = y
      A,B,C = y

      # dDdt = k3*(D_init - G) - (k0 + k3)*D
      # dGdt = (D_init - D)*(k1 + k2*G) - G*(k1 + k4) - G**2*(k2 + k5)
      dAdt = B*(k4 - k2*A + k5*B) - k1*A
      dCdt = k0*B - k3*C
      dBdt = -1*(dAdt+dCdt)

      return [dAdt, dBdt, dCdt]

    # default 'RK45', alt 'BDF' 'RK23' 'DOP853' 'Radau' 'LSODA'
    sol = solve_ivp(rhs, t_span, y0 = [A_init,B_init,C_init],
        dense_output=True, method=self.method)

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    Y = sol.sol(t)
    return fl_A*Y[0] + fl_B*Y[1] + fl_C*Y[2]

################################################################################################################################

#@title Boateng1 Model v.2_partial
#@markdown

class Boateng1Model_v2(CRNModel) :

  model_cnt = 0
  method = 'RK45'

  def __init__(self,
               t_eval, emp_data,  # super
               name=None,         # super
               method='RK45') :   # self

    self.method = method

    # argument validation
    assert is_list_like(t_eval) , f'{type(self)}.__init__ argument \'t_eval\' must be list-like.'
    assert is_list_like(emp_data) , f'{type(self)}.__init__ argument \'emp_data\' must be list-like.'

    if not name :
      name = f'Boateng1Model_v2_{model_cnt}'
      model_cnt+=1

    # Set empirical limits on Parameters
    # (name, value=None, vary=True, min=-inf, max=inf, expr=None, brute_step=None)
    emp_params = Parameters()
    emp_params.add('k0', value=1, vary=True, min=0) # 2nd aggregation
    emp_params.add('k1', value=1, min=0) # nucleation
    emp_params.add('k2', value=1, min=0) # 1st aggregation
    emp_params.add('k3', value=1, vary=True, min=0) # rev. 2nd aggr
    emp_params.add('k4', value=0, vary=True, min=0) # rev. nucl
    emp_params.add('k5', value=0, vary=True, min=0) # rev. aggr
    emp_params.add('fl_A', value=1.0, min=0.01) # fl_A
    emp_params.add('fl_B', value=1.0, min=0.01) # fl_B
    emp_params.add('fl_C', value=1.0, min=0.01) # fl_C
    emp_params.add('A_init', value=1, min=0.01) # initial conc. A
    emp_params.add('B_init', value=0.1, min=0.01) # initial conc. B
    # emp_params.add('C_init', value=0.1, vary=True, min=0.01) # initial conc. C

    # super constructor
    super().__init__(t_eval = t_eval,
                     emp_data = emp_data,
                     emp_params = emp_params,
                     name = name)

  def func(self,t, k0,k1,k2,k3,k4,k5, fl_A,fl_B,fl_C, A_init,B_init):
    t = np.asarray(t)
    t_span = (t[0], t[-1])

    def rhs(T, y):
      # D,G = y
      A,B,C = y

      # dDdt = k3*(D_init - G) - (k0 + k3)*D
      # dGdt = (D_init - D)*(k1 + k2*G) - G*(k1 + k4) - G**2*(k2 + k5)
      dAdt = B*(k4 - k2*A + k5*B) - k1*A
      dCdt = k0*B - k3*C
      dBdt = -1*(dAdt+dCdt)

      return [dAdt, dBdt, dCdt]

    # default 'RK45', alt 'BDF' 'RK23' 'DOP853' 'Radau' 'LSODA'
    sol = solve_ivp(rhs, t_span, y0 = [A_init,B_init,0],
        dense_output=True, method=self.method)

    if not sol.success:
        raise RuntimeError("ODE solver failed: " + sol.message)

    Y = sol.sol(t)
    return fl_A*Y[0] + fl_B*Y[1] + fl_C*Y[2]

################################################################################################################################

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
