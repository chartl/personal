import numpy as np
from cp_lrt_utils import *
import cp_lrt_simulations
import cp_permute
import time
import warnings
warnings.simplefilter('error')

from multiprocessing import Pool

MP_CHUNK_SIZE=5000

def par_perm(Y, Xp, sample_ids=None):
  if sample_ids is None:
    p = np.random.permutation(np.arange(Y.shape[1]))
  else:
    q = list()
    for sid in np.unique(sample_ids):
      six = np.where([s == sid for s in sample_ids])[0]
      q.append(six)
    qp = np.random.permutation(q)  # permute the list of indeces
    p = [i for six in qp for i in six]
  Xp[0,:] = Xp[0,p]
  permuted_b = np.linalg.lstsq(Xp.T, Y.T, rcond=None)[0][0,:]
  return np.sum(permuted_b)


def par_perm_(args):
  return par_perm(*args)


def do_permutation(Y, x, C, perm_args, pool, sample_ids=None):
    # special case no variability in X
    if np.var(x) == 0:
      return {'Bsum': np.nan, 'p': 1, 'cns': np.nan, 'n_perm': 0, 'n_gene': Y.shape[0],
              'n_sample': Y.shape[1], 'normal': np.nan, 'p_perm': 1.0}
    X = np.vstack([x, C])
    observed_b = np.linalg.lstsq(X.T, Y.T, rcond=None)[0][0,:]
    observed_L = np.sum(observed_b)
    n_higher_permuted_L = 0
    Lvec = []
    Xp = None
    if pool is None:
      Xp = Xp or X.copy()
      tstart = time.time()
      for p in xrange(1, 1 + perm_args['maxperm']):
        if p % perm_args['checkperm'] == 0 and p > 0:
          conf_nsig = cp_permute.calc_conf_nsig(p, n_higher_permuted_L, perm_args['signif_threshold'])
          elapsed = time.time() - tstart
          tpp = elapsed/p  # seconds per iter
          tpe = perm_args['checkperm'] * tpp
          tte = (perm_args['maxperm'] - p) * tpp
          tteh = tte/(60*60)
          _, normal_test = sp.stats.normaltest(Lvec)
          print('perm: %d, ns_conf: %.2e, epctm(s): %.1f, rem(h)[%d]: %.1f, nnp: %.2e' % (p, conf_nsig, tpe, perm_args['maxperm'] - p, tteh, normal_test))
          if conf_nsig > perm_args['conf_nsig']:
            break
        permuted_b = par_perm(Y, Xp, sample_ids)
        permuted_L = np.sum(permuted_b)
        if np.abs(permuted_L) >= np.abs(observed_L):
          n_higher_permuted_L += 1
        Lvec.append(permuted_L)
    else:
      chunk_size = perm_args.get('checkperm', MP_CHUNK_SIZE)
      num_chunks = int(perm_args['maxperm']/chunk_size) + 1
      Lvec = []
      tstart = time.time()
      for chnk in xrange(1, 1 + num_chunks):
        data = ((Y, X, sample_ids) for _ in xrange(chunk_size))
        Ls = pool.map(par_perm_, data)
        n_higher_permuted_L += np.sum([np.abs(L) >= np.abs(observed_L) for L in Ls])
        conf_nsig = cp_permute.calc_conf_nsig(chnk * chunk_size, n_higher_permuted_L, perm_args['signif_threshold'])
        elapsed = time.time() - tstart
        tpp = elapsed/p  # seconds per iter
        tpe = perm_args['checkperm'] * tpp
        tte = (perm_args['maxperm'] - p) * tpp
        tteh = tte/(60*60)
        _, normal_test = sp.stats.normaltest(Lvec)
        print('perm: %d, ns_conf: %.2e, epctm(s): %.1f, rem(h)[%d]: %.1f, nnp: %.2e' % (p, conf_nsig, tpe, perm_args['maxperm'] - p, tteh, normal_test))
        if conf_nsig > perm_args['conf_nsig']:
            break
        print(' .. permutation {}, nonsig_conf: {}, sig: {}'.format(chunk_size * chnk, conf_nsig, n_higher_permuted_L))
        Lvec.extend(Ls)
      p = chnk * chunk_size
      
    results = {'Bsum': observed_L, 'p_perm': float(n_higher_permuted_L)/p,
               'cns': conf_nsig, 'n_perm': p, 'n_gene': Y.shape[0],
               'n_sample': Y.shape[1], 'p': np.nan, 'normal': np.nan} # 'B': ','.join(map(str, observed_b))}
    if perm_args['normal_approx']:
      # use a monotonic transformation
      try:
        Lvstar, lmb = sp.stats.yeojohnson(Lvec)
        observed_Lstar = sp.stats.yeojohnson(observed_L, lmb)
        params = list(sp.stats.t.fit(Lvstar))
	delta = observed_Lstar - params[1]
      except RuntimeWarning:
        params = list(sp.stats.t.fit(Lvec))
        Lvstar = Lvec
        delta = 0.
      #params = list(sp.stats.t.fit(Lvec))
      #delta = observed_L - params[1]
      #trans_z = sp.stats.norm.ppf(sp.stats.t.sf(Lvec, *params))
      _, normal_test = sp.stats.normaltest(Lvstar)
      params[1] = 0.
      approx_p = 2 * (1. - sp.stats.t.cdf(np.abs(delta), *params))
      results['p'] = approx_p
      results['normal'] = normal_test
    return results

def run_permutation(args):
    Y, X, C, G, fit_args, reg_args = parse_args(args)
    rsids = X.index
    Y, X, C, G = Y.values, X.values, C.values, G.values
    Y = (Y - np.mean(Y, axis=1)[:, np.newaxis])/np.std(Y, axis=1)[:, np.newaxis]
    grm_pcs = sp.sparse.linalg.svds(G, k=4, which='LM', return_singular_vectors=True)[0]
    covars = np.hstack([C, grm_pcs]).T
    perm_args = cp_permute.get_perm_args(args)
    perm_args['n_cores'] = args.n_cores
    pool = None
    if perm_args['n_cores'] > 1:
        pool = Pool(perm_args['n_cores'])
    fit_results = list()
    if args.var_start is not None and args.var_end is not None:
        iter_ = xrange(args.var_start, args.var_end)
        rsids = rsids[args.var_start:args.var_end]
    elif args.var_start is not None:
        iter_ = xrange(args.var_start, X.shape[0])
        rsids = rsids[args.var_start:]
    else:
        iter_ = xrange(X.shape[0])
    for i in iter_:
        print('Variant {}'.format(i+1))
        presult = do_permutation(Y, X[(i,),:], covars, perm_args, pool)
        fit_results.append(presult)
    fit_results = pd.DataFrame(fit_results)
    fit_results['variant'] = [x for x in rsids]
    fit_results.to_csv(args.output)


def run_simulation(args):
    results = list()
    fit_args = {k: getattr(args, k) for k in DEFAULT_FIT_ARGS}
    if args.simulation_stop is None:
      sstop = 1 + args.simulation
    else:
      sstop = args.simulation_stop
    permargs = cp_permute.get_perm_args(args)
    permargs['n_cores'] = args.n_cores
    for sim in xrange(args.simulation, sstop):
      for _ in xrange(args.sim_replicates):
        data, info = cp_lrt_simulations.run_single_sim(sim, args.sim_param_file)
        grm_pcs = sp.sparse.linalg.svds(data['G'].values, k=4, which='LM', return_singular_vectors=True)[0]
        covars = np.hstack([data['C'].values.T, grm_pcs]).T
        results.append(do_permutation(data['Y'].values, data['x'].values, covars, permargs))
    results = pd.DataFrame(results)
    for param, param_val in info.items():
        results[param] = param_val
    results.to_csv(args.output)


def do_normal_approx(Y, x, C):
  """\
  We have the assumption here that (using X = [ - x - ]
                                              [ - C - ])

  (g x s) ~ MN [ (g x f) (f x s), (g x g), (s x s)]
  Y ~ MN(BX, Eg, K)

  solving for B:
  (g x s) (s x f) (f x f)
  Y X.T * inv(X X.T) =

                (f x f)   (f x s) (s x s) (s x f) (f x f) 
  Bhat ~ MN(B, Eg, inv(X X.T) * X * K * X.T * inv(X X.T))

  We care only about B[x] -- or the first row -- and this would look like

  B[x] = B[0, :] ~ N(B[x], Eg * [inv(X X.T) * X * K * X.T * inv(X X.T)]_{1,1})
  # ~ N(B[x], Eg * const)

  # if K = I, then const is just
  # const = inv(X X.T)[1,1]

  # finally
  # sum(Bhat[0,:]) ~ N(sum(B[x]), sum(E) * const)


  """
  X = np.vstack([x, C])
  observed_B = np.linalg.lstsq(X.T, Y.T, rcond=None)[0]
  observed_b = observed_B[0, :]
  observed_L = np.sum(observed_b)
  residuals = Y - np.dot(observed_B.T, X)
  resid_covar = np.cov(residuals)
  try:
    column_var = np.linalg.inv(np.dot(X, X.T))[0, 0]
  except np.linalg.linalg.LinAlgError:
    column_var = np.linalg.inv(np.dot(X, X.T) + np.diag(1e-4 * np.ones((X.shape[0],))))[0, 0]
  observed_b_variance = np.sum(resid_covar) * column_var
  Z = observed_L / np.sqrt(observed_b_variance)
  p = 2 * (1. - sp.stats.norm.cdf(np.abs(Z), loc=0., scale=1.))
  results = {'Bsum': observed_L, 'p_perm': 'NA',
              'cns': 'NA', 'n_perm': 'NA', 'n_gene': Y.shape[0],
              'n_sample': Y.shape[1], 'p': p, 'normal': np.nan}
  return results



def main(args):
  assert args.output is not None
  if args.simulation > 0:
    run_simulation(args)
  else:
    run_permutation(args)


if __name__ == '__main__':
  parser = cp_permute.mkparser()
  parser.add_argument('--simulation_stop', help='stop at this simulation', type=int, default=None)
  parser.add_argument('--sim_param_file', help='File for simulation parameters', type=str, default=None)
  parser.add_argument('--n_cores', help='Number of cores for parallel', type=int, default=1)
  parser.add_argument('--var_start', help='Variant at which to start', type=int, default=None)
  parser.add_argument('--var_end', help='Variant at which to end', type=int, default=None)

  args = parser.parse_args()
  main(args)

