import numpy as np
import pandas as pd

import pgmpy
from pgmpy.inference.ExactInference import VariableElimination



def var_elim(dbn, vars, ev):
    ve = VariableElimination(dbn)
    disf = ve.query(variables=vars,
                                 evidence=ev)
    inds = pd.DataFrame(np.argwhere(disf.values), 
                 columns = disf.variables)
    vals = disf.values[np.nonzero(disf.values)]
    for i in range(inds.shape[1]):
        varb= disf.variables[i]
        inds.iloc[:,i] = [disf.state_names[varb][x] for x in inds.iloc[:,i]]
    inds = pd.DataFrame(vals, 
                        index = pd.MultiIndex.from_frame(inds),
                       columns = ['proportion'])
    summ = inds
    return summ



def map_to_fc(expr):
    if expr == (0,0,0,0,0):
        return '$C^-$'
    elif expr[2] == 1:
        return '$C^+_{16L}$'
    elif expr[3] == 1:
        return '$C^+_{32E}$'
    elif expr[4] == 1:
        return '$C^+_{32L}$'

def map_to_final_C(expr):
    if expr[4] == 0:
        return '$C^-$'
    else:
        return '$C^+$'

def map_to_final_S(expr):
    if expr[4] == 0:
        return '$C^-$'
    else:
        return '$C^+$'

def map_to_fs(expr):
    if expr == (0,0,0,0,0):
        return '$S^-$'
    elif expr[2] == 1:
        return '$S^+_{16L}$'
    elif expr[3] == 1:
        return '$S^+_{32E}$'
    elif expr[4] == 1:
        return '$S^+_{32L}$'

def map_to_class(v):
    x = v[0]
    y = v[1]
    if x == 0:
        if y == 0:
            return '$C^-S^-$'
        else:
            return '$C^-S^+$'
    else:
        if y == 1:
            return '$C^+S^+$'
        else:
            return '$C^+S^-$'
