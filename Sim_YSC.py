import pgmpy

from pgmpy.models import BayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from sklearn.utils import resample

import numpy as np
import pandas as pd

def simulate_tree(dbn = None, var_names = None):
    states = {}
    origin = dbn.simulate(1, show_progress = False)[[v+'0' for v in var_names]]
    num_cells = [2,1,2,1]
    states[0] = origin
    for j in range(4):
        num_cell = num_cells[j]
        prev_vars = [v+str(j) for v in var_names]
        next_vars = [v+str(j+1) for v in var_names]
        daughters = []
        for st in origin.values:
            daughters.append(dbn.simulate(num_cell, {prev_vars[i]: np.ravel(st)[i] for i in range(len(var_names))}, show_progress = False)[next_vars])
        daughters = pd.concat(daughters)
        states[j+1] = daughters
        origin = daughters
    return states
    
def concatenate_tree(states):
    n_reps = [4,2,2,1,1]
    branches = pd.concat([pd.DataFrame(np.repeat(states[i].values, n_reps[i], axis = 0), columns = states[i].columns) for i in range(5)], axis = 1)
    index= []
    for i in range(5):
        index.append(list(np.repeat(states[i].index.astype(str), n_reps[i])))
    result = [''.join(items) for items in zip(*index)]
    branches.index = result
    return branches
    
def simulate_embryo(dbn = None, var_names = None):
    trees = {}
    for j in range(8):
        trees[j] = simulate_tree(dbn, var_names = var_names)
    return trees
    
def concatenate_embryo(cells):
    cemb = pd.concat([concatenate_tree(cells[i]) for i in range(8)], axis = 0, keys = range(8))
    cemb.index.names = ['Lineage', 'Code']
    return cemb


def bootstrap_network(df, win_edg = None, num_samples=10, estimator_type="mle", **estimator_kwargs):
    """
    Learns a Bayesian network using bootstrap resampling.

    Args:
        data (pd.DataFrame): The input data.
        num_samples (int): The number of bootstrap samples.
        estimator_type (str): "mle" for Maximum Likelihood Estimation or "bayes" for Bayesian Estimation.
        **estimator_kwargs: Additional keyword arguments for the chosen estimator.

    Returns:
        list of pgmpy.BayesianNetwork or list of pgmpy.CPD: A list of networks (or CPTs if estimator is Bayesian) and associated
        CPTs
    """

    networks = []
    all_cpts = []
    for _ in range(num_samples):
        resampled_data = resample(df, replace=True)
        model = BayesianNetwork()
        model.add_edges_from(win_edg)
        if estimator_type == "mle":
            estimator = MaximumLikelihoodEstimator
            model.fit(resampled_data, estimator=estimator)
            networks.append(model)
            all_cpts.append(model.get_cpds())

        elif estimator_type == "bayes":
            estimator = BayesianEstimator
            # Calculate posteriors for CPTs
            posteriors = estimator.get_parameters(prior_type='dirichlet', pseudo_count=1)
            all_cpts.append(posteriors)
            networks.append(model)
        else:
            raise ValueError("Estimator type must be 'mle' or 'bayes'")
    return networks, all_cpts

def simulate_once(df, win_edg = None, var_names = None):
    nets, _ = bootstrap_network(df, win_edg = win_edg, num_samples = 1)
    emb = simulate_embryo(nets[0], var_names = var_names)
    return emb
