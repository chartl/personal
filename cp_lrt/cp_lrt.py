from argparse import ArgumentParser
import scipy as sp
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats

from cp_lrt_utils import *
from cp_lrt_tensorflow import *
import cp_fixedperm

DEFAULT_REGULARIZATION = {
  'B': (0.05, None),
  'D': (0.05, None),
  'L': (0.01, None)
}

IGNORE_CONVERGENCE=True


def fit_single_variant_restart(Y, x, C, G, grank=3, init_vals=None, lambda_reg=None, fit_args=None, return_all_params=False, verbose=0, nr=2, ignore_convergence=IGNORE_CONVERGENCE):
    """\
    Fit the model `nr` times, taking the maximum likelihood across all fits for constrained and unconstrained models

    """
    best_constrained_fit, best_unconstrained_fit = None, None
    converged_fits = 0
    for try_ in xrange(2*nr):
        fit_result = fit_single_variant(Y, x, C, G, grank, init_vals, lambda_reg, fit_args, return_all_params, verbose)
        if not (fit_result[2] and fit_result[3]):
            continue
        converged_fits += 1
        if best_constrained_fit is None:
            best_constrained_fit, best_unconstrained_fit = fit_result, fit_result
        else:
            if best_constrained_fit[1] < fit_result[1]:
                best_constrained_fit = fit_result
            if best_unconstrained_fit[0] < fit_result[0]:
                best_unconstrained_fit = fit_result
        if converged_fits >= nr:
            break
    else:
        if ignore_convergence:
            print ('Variant failed to converged after {} tries. Consider altering fit parameters.'.format(2*nr))
        else:
            raise ValueError('Variant failed to converged after {} tries. Consider altering fit parameters.'.format(2*nr))
    return (best_unconstrained_fit[0], best_constrained_fit[1],
            best_unconstrained_fit[2], best_constrained_fit[3],
            best_unconstrained_fit[4], best_unconstrained_fit[5])
     

def fit_single_variant(Y, x, C, G, grank=3, init_vals=None, lambda_reg=None, fit_args=None, return_all_params=False, verbose=0):
    """\
    Use tensorflow+ADAM to fit unconstrained and constrained ML parameters for the model

    Y ~ MN(Bx + DC, E, kG + sI)

    :input Y:          the data matrix (pheno x sample) to fit (my use: gene x sample)
    :input x:          the dosages for a single variant (1 x sample)
    :input C:          the sample covariates (covar x sample)
    :input G:          the genetic relationship matrix (sample x sample)
    :input grank:      the rank of the low-rank perturbative approximation to gene-gene covariance
    :input init_vals:  initial values for parameters, as a matrix. See `cp_lrt_tensorflow` for details.
                         `B` - the (pheno x 1) matrix of variant coefficients
                         `D` - the (pheno x covar) matrix of covariate effects
                         `K` - the (gene x 1) vector of gene variances
                         `L` - the (gene x `grank`) matrix of gene covariance perturbations
                         `k` - the (1 x 1) scalar of genetic effects
    :input lambda_reg: Regularization factors for any parameters, given as a {paramname: coefficient} dict
    :input fit_args:   Arguments for the fit (maxiter, learning rate, and convergence; see DEFAULT_FIT_DESC)
    :input return_all_params: whether to return the final values of all parameters
    :input verbose:    how noisy to be

    """
    print([t.shape for t in (Y, x, C, G)])
    fit_args = fit_args or DEFAULT_FIT_ARGS
    model, model_vars = build_model(Y, x, C, grank, G, init_vars=init_vals)
    fit_model_uc, model_params_uc = fit_model(Y, model, model_vars, fit_args, lambda_reg)
    print('Unconstrained beta sum: {}'.format(model_params_uc['B_sum']))
    modelC, modelC_vars = build_model(Y, x, C, grank, G, init_vars=model_params_uc, constrain_B=True)
    fit_model_c, model_params_c = fit_model(Y, modelC, modelC_vars, fit_args, lambda_reg)
    print 'Constrained beta sum: {}'.format(np.sum(model_params_c['B_sum']))
    all_params = {'unconstrained': model_params_uc, 'constrained': model_params_c} if return_all_params else None
    return (model_params_uc['lik'], model_params_c['lik'], 
            model_params_uc['converged'], model_params_c['converged'],
            np.sum(model_params_uc['B']), all_params)
    

def main(args):
    Y, X, C, G, fit_args, reg_args = parse_args(args)
    if reg_args is None and not args.no_regularization:
      reg_args = DEFAULT_REGULARIZATION
    #Y, X, C = scale_matrices(Y, X, C)
    Xi, Xc = X.index, X.columns
    rsids = Xi
    sample_ids = np.array([x for x in X.columns]) if args.within_samples else None
    Y, X, C, G = Y.values, X.values, C.values, G.values
    if args.samples:
        sample_ids = [x.strip() for x in open(args.samples)]
    Gsvd = sp.sparse.linalg.svds(G, k=4, which='LM', return_singular_vectors=True)[0]
    Cperm = np.hstack([C.T, Gsvd]).T
    if np.linalg.cond(Cperm) > 10e15:
        raise ValueError('Ill-conditioned covariates. Check covariate matrix.')
    if args.var_start is not None and args.var_end is not None:
        iter_ = xrange(args.var_start, args.var_end)
        rsids = rsids[args.var_start:args.var_end]
    elif args.var_start is not None:
        iter_ = xrange(args.var_start, X.shape[0])
        rsids = rsids[args.var_start:]
    else:
        iter_ = xrange(X.shape[0])
    fit_results = list()
    perm_args = {'conf_nsig': 0.9,
                 'maxperm': args.permute,
                 'checkperm': 1000,
                 'signif_threshold': 0.01,
                 'n_cores': 1,
                 'normal_approx': True}
    for i in iter_:
        if i % 50 == 0:
          print('.. iter {} / {}'.format(i,args.var_end))
        # step 1: use the normal approximation to get a quick sense of significance
        approx_res = cp_fixedperm.do_normal_approx(Y, X[(i,),:], Cperm)
        if (not np.isnan(approx_res['p']) and approx_res['p'] > args.permutation_threshold) and sample_ids is None:
            results = (np.nan, np.nan, 3, 3, approx_res['Bsum'], approx_res['p'], np.nan, np.nan, np.nan)
        else:
            # step 2: use a permutation to get a sense of normality
            permres = cp_fixedperm.do_permutation(Y, X[(i,),:], Cperm, perm_args, pool=None, sample_ids=sample_ids)
            if args.model_threshold < 0 or permres['normal'] > args.model_threshold:
                results = (np.nan, np.nan, 2, 2, permres['Bsum'], approx_res['p'], permres['normal'], permres['p'], permres['p_perm'])
            else:
                # step 3: the normal approximation is significant, the permutation suggests the distribution is not normal
                # so we fall back to the full likelihood ratio
                model_res = fit_single_variant_restart(Y, X[(i,), :], C, G, args.lra_rank, None, reg_args, fit_args, return_all_params=False, nr=2)[:5]
                results = list(model_res) + [approx_res['p'], permres['normal'], permres['p']]
        fit_results.append(results)
    print('Converting to dataframe')
    fit_results = pd.DataFrame(fit_results)
    fit_results.columns = ['lik', 'lik_cons', 'conv', 'conv_con', 'lambda_lap', 'p_norm_approx', 'p_isnormal', 'p_perm_approx', 'p_perm']
    print('dealing with chisq')
    idx = np.where([pd.notna(fit_results.lik)])[0]
    fit_results['p_chisq'] = np.nan
    fit_results['log_lik_ratio'] = np.nan
    if len(idx) > 0:
      fit_results['log_lik_ratio'].values[idx] = fit_results.lik.values[idx] - fit_results.lik_cons.values[idx]
      fit_results['p_chisq'].values[idx] = sp.stats.chi2.sf(2 * fit_results.log_lik_ratio.values[idx], df=1)
    print('changing index')
    fit_results.index = rsids
    print('writing')
    fit_results.to_csv(args.output, sep='\t', na_rep='NA')



if __name__ == '__main__':
    parser = mkparser()
    parser.add_argument('--permute', help='Use this many permutations only', type=int, default=20000)
    parser.add_argument('--no_regularization', help='No default regularization', action='store_true')
    parser.add_argument('--permutation_threshold', help='Switch from approx to permutation at this significance', type=float, default=0.0001)
    parser.add_argument('--model_threshold', help='Switch from permutation to full model at this (goodness-of-fit) significance', type=float, default=0.01)
    parser.add_argument('--within_samples', help='Perform permutation within samples (repeated samples)', action='store_true')
    parser.add_argument('--samples', help='file listing the samples one per line', default=None)
    args = parser.parse_args()
    print(args)
    main(args)

