import pandas as pd
import scipy as sp
import numpy as np
from argparse import ArgumentParser

DEFAULT_FIT_ARGS = {'nesterov': 5e-4, 'nesterov_final': 1e-12, 'progress_iter': 250, 'maxiter': 16 * 5000,
                    'param_convergence_tol': 5e-3, 'loss_convergence_tol': 1e-3,
                    'nesterov_power': 0.5, 'cycleiter': 10000}
DEFAULT_FIT_DESC = {'maxiter': 'maximum iterations',
                    'nesterov': 'learning rate for ADAM optimizer', 'progress_iter': 'Print progress with this period',
                    'param_convergence_tol': 'Define convergence as a maximum gradient norm of THIS, over aparameters',
                    'loss_convergence_tol': 'Define convergence as this change in likelihood over a step; must ALSO converge in parameters',
                    'nesterov_final': 'The final learning rate',
                    'nesterov_power': 'The power for the learning rate scheduler',
                    'cycleiter': 'Iterations to cycle the learning rate scheduler'}


def parse_args(args):
    fit_args = DEFAULT_FIT_ARGS
    Y = pd.read_csv(args.expression, sep='\t', index_col=0)
    X = pd.read_csv(args.dosages, sep='\t', index_col=0)
    if Y.shape[1] != X.shape[1]:
       Y = Y.T
    if Y.shape[1] != X.shape[1] or not all([a == b for a, b in zip(Y.columns, X.columns)]):
      msg = 'Column mismatch:  Y=\n'
      msg += ','.join(Y.columns) + '\n\nX=\n'
      msg += ','.join(X.columns)
      raise ValueError(msg)
    C = pd.read_csv(args.covariates, sep='\t', index_col=0)
    if 'grm' in dir(args):
        G = pd.read_csv(args.grm, sep='\t', index_col=0)
        assert G.shape[1] == X.shape[1], repr([G.shape, X.shape])
        assert G.shape[0] == G.shape[1], G.shape
    else:
        G = None
    for k in DEFAULT_FIT_ARGS:
        if k in dir(args):
          fit_args[k] = getattr(args, k)
    if 'regularization' in dir(args) and args.regularization:
        reg_args = dict()
        for pair in args.regularization.split(','):
            var, val = pair.split('=')
            reg_args[var] = (float(val), 0.0)  # target is always 0
    else:
        reg_args = None
    return Y, X, C, G, fit_args, reg_args


def mkparser():
    parser = ArgumentParser('cp_lrt')
    parser.add_argument('expression', help='The gene expression matrix to test for consistent effect directions', type=str)
    parser.add_argument('dosages', help='The genotype dosage matrix', type=str)
    parser.add_argument('covariates', help='The covariate matrix', type=str)
    parser.add_argument('grm', help='The GRM', type=str)
    parser.add_argument('output', help='The output file', type=str)
    parser.add_argument('--lra_rank', help='Rank of the low-rank perturbation (gene-gene covariance)', type=int, default=3)
    parser.add_argument('--var_start', help='Variant at which to start', type=int, default=None)
    parser.add_argument('--var_end', help='Variant at which to end', type=int, default=None)
    parser.add_argument('--regularization', help='Regularization string (variable=value,variable=value,...)', default=None)
    for defarg in DEFAULT_FIT_ARGS:
        aname = '--' + defarg
        parser.add_argument(aname, help=DEFAULT_FIT_DESC[defarg], default=DEFAULT_FIT_ARGS[defarg], type=DEFAULT_FIT_ARGS[defarg].__class__)
    return parser
