"""\
Permutation alternative to the LRT

"""
from cp_lrt_utils import *
import cp_lrt_simulations 

def mkparser():
    parser = ArgumentParser('cp_permute')
    parser.add_argument('--expression', help='The gene expression matrix to test for consistent effect directions', type=str, default=None)
    parser.add_argument('--dosages', help='The genotype dosage matrix', type=str, default=None)
    parser.add_argument('--covariates', help='The covariate matrix', type=str, default=None)
    parser.add_argument('--grm', help='The GRM file', type=str, default=None)
    parser.add_argument('--output', help='The output file', type=str, default=None)
    parser.add_argument('--simulation', help='Run this simulation from cp_lrt_simulations', type=int, default=-1)
    parser.add_argument('--sim_replicates', help='The number of replicates of this simulation to run', default=1, type=int)
    parser.add_argument('--conf_nsig', help='Stop permuting when confidence in non-significance exceeds this value', type=float, default=0.9999)
    parser.add_argument('--maxperm', help='Maximum permutations', type=int, default=100000000)
    parser.add_argument('--signif_threshold', help='The significance threshold', type=float, default=0.01)
    for defarg in DEFAULT_FIT_ARGS:
        aname = '--' + defarg
        parser.add_argument(aname, help=DEFAULT_FIT_DESC[defarg], default=DEFAULT_FIT_ARGS[defarg], type=DEFAULT_FIT_ARGS[defarg].__class__)
    return parser


def calc_conf_nsig(nsim, nhigher, thresh):
    # use a beta-binomial to estimate the probability of the
    # 'true' p-value falling above the threshold
    a_prior = 5.
    b_prior = 5000.
    a_post = a_prior + nhigher
    b_post = b_prior + (nsim - nhigher)
    return sp.stats.beta.sf(thresh, a_post, b_post)


def fit_genes(Y, X, C, G, model, mvars, init_args, quick_args):
    bvals = list()
    for row in xrange(Y.shape[0]):
      _, fit_vals = fit_model(Y[(row,),:], model, mvars, quick_args, verbose=False)
      if fit_vals['converged'] == 0:
        _, fit_vals = fit_model(Y[(row,),:], model, mvars, init_args)
      bvals.append(fit_vals['B'])
    return bvals

def do_permutation(Ymat, Xmat, Cmat, Gmat, perm_args, regularizers=None, fit_args=None):
    fit_args = DEFAULT_FIT_ARGS if fit_args is None else fit_args
    print(fit_args)
    # start with observed, independent parameters
    model, mvars = build_model_singlegene(Ymat.values[(0,),:], Xmat.values[(0,),:], Cmat, Gmat)
    quick_args = {k: v for k, v in fit_args.items()}
    quick_args['maxiter'] = 1000
    quick_args['nesterov'] = 5e-2
    quick_args['nesterov_final'] = 1e-10
    quick_args['progress_iter'] = 5000
    observed_b = fit_genes(Ymat.values, Xmat.values, Cmat.values, Gmat.values, model, mvars, fit_args, quick_args)
    observed_L = np.sum(observed_b)
    n_higher_permuted_L = 0
    for p in xrange(perm_args['maxperm']):
        if p % 25 == 0 and p > 0:
          conf_nsig = calc_conf_nsig(p + 1, n_higher_permuted_L, perm_args['signif_threshold'])
          if conf_nsig > perm_args['conf_nsig']:
              break
          print('permutation {}: nonsig_conf: {}'.format(p, conf_nsig))
        perm = np.random.permutation(np.arange(Y.shape[1]))
        X = Xmat.iloc[:, perm]
        C = Cmat.iloc[:, perm]
        G = Gmat.iloc[perm, :]
        G = G.iloc[:, perm]
        model, mvars = build_model_singlegene(Ymat.values[0,:], X.values[0,:], C, G)
        permuted_b = fit_genes(Ymat.values, X.values, C.values, G.values, model, mvars, fit_args, quick_args)
        permuted_L = np.sum(permuted_b)
        if np.abs(permuted_L) > np.abs(observed_L):
            n_higher_permuted_L += 1
    return {'Bsum': observed_L, 'p': float(n_higher_permuted_L)/p,
            'cns': conf_nsig, 'n_perm': p, 'n_gene': Y.shape[0], 'B': ','.join(map(str, observed_b))}


def get_perm_args(args):
    pargs = dict()
    for k in ['conf_nsig', 'maxperm', 'signif_threshold']:
        pargs[k] = getattr(args, k)
    return pargs


def run_simulation(args):
    results = list()
    fit_args = {k: getattr(args, k) for k in DEFAULT_FIT_ARGS}
    print('on input:')
    print(fit_args)
    for _ in xrange(args.sim_replicates):
        data, info = cp_lrt_simulations.run_single_sim(args.simulation)
        results.append(do_permutation(data['Y'], data['x'], data['C'], data['G'], get_perm_args(args), fit_args=fit_args))
    results = pd.DataFrame(results)
    for param, param_val in info.items():
       results[param] = param_val
    results.to_csv(args.output)



def run_permutation(args):
    Y, X, C, G, fit_args, reg_args = parse_args(args)
    perm_args = get_perm_args(args)
    fit_results = list()
    for i in range(X.shape[0]):
        presult = do_permutation(Y, X[(i,),:], C, G, perm_args, reg_args, fit_args)
        fit_results.append(presult)
    fit_results = pd.DataFrame(fit_results)
    fit_results.to_csv(args.output)


def main(args):
    assert args.output is not None
    if args.simulation > 0:
        run_simulation(args)
    else:
        run_permutation(args)


if __name__ == '__main__':
    parser = mkparser()
    args = parser.parse_args()
    main(args)
