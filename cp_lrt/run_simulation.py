import cp_lrt_simulations
import cp_lrt
import cp_fixedperm
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats

from argparse import ArgumentParser
parser = ArgumentParser('run_simulation')
parser.add_argument('sim_number', type=int)
parser.add_argument('--n_replicates', type=int, default=4)
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--sim_offset', type=int, default=0, help='Offset the simulation number')
parser.add_argument('--params', type=str, default=None, help='the param file')
parser.add_argument('--permute', type=int, default=None, help='Run a permutation with this many iterations instead of LR test')
parser.add_argument('--norm_approx', action='store_true', help='Use the normal approximation')

args = parser.parse_args()

if args.output is None:
  output = 'simulation_{}.results.txt'.format(args.sim_number)
else:
  output = args.output

results = list()
B_REG_VEC = [0.] if args.permute else [0., 0.5]
for replicate in xrange(args.n_replicates):
  data, info = cp_lrt_simulations.run_single_sim(args.sim_number, args.params)
  for fit_rank in [int(q * info['n_samples']) for q in [0.01]]:
    for B_reg_alpha in B_REG_VEC:
      B_reg = B_reg_alpha * info['n_genes'] * max(0, ((info['n_genes'] - info['n_samples'])/info['n_samples']))
      for fit_repl in xrange(2):
        print(' --- this run: B_lambda = {} ---'.format(B_reg))
        if args.norm_approx:
          Gsvd = sp.sparse.linalg.svds(data['G'], k=10, which='LM', return_singular_vectors=True)[0]
          Cperm = np.hstack([data['C'].T, Gsvd]).T
          normres = cp_fixedperm.do_normal_approx(data['Y'].values, data['x'].values.reshape(info['n_samples'],), Cperm)
          print(normres)
          fit_results = [np.nan, np.nan, 3, 3, normres['Bsum'], normres['p_perm'], normres['cns'], normres['p']]
        elif args.permute is not None:
          perm_args = {'conf_nsig': 0.9,
                       'maxperm': args.permute,
                       'checkperm': 5000,
                       'signif_threshold': 0.01,
                       'n_cores': 1,
                       'normal_approx': True}
          Gsvd = sp.sparse.linalg.svds(data['G'], k=10, which='LM', return_singular_vectors=True)[0]
          Cperm = np.hstack([data['C'].T, Gsvd]).T
          permres = cp_fixedperm.do_permutation(data['Y'].values, data['x'].values.reshape(info['n_samples'],), Cperm, perm_args, None)
          print(permres)
          fit_results = [np.nan, np.nan, 2, 2, permres['Bsum'], permres['p_perm'], permres['cns'], permres['p']]
        else:
          fit_results = cp_lrt.fit_single_variant(data['Y'], data['x'], data['C'], data['G'], fit_rank, lambda_reg={'B': (B_reg, None)})[:5]
        results.append([t for t in fit_results] + [fit_rank, B_reg, replicate, fit_repl])

fit_results = pd.DataFrame(results)
if args.permute is None and not args.norm_approx:
  fit_results.columns = ['lik', 'lik_cons', 'conv', 'conv_con', 'lambda_lap',
                         'L_rank', 'B_lambda', 'replicate', 'fit_replicate']
  fit_results['log_lik_ratio'] = fit_results.lik - fit_results.lik_cons
  fit_results['log_lik_ratio_chisq'] = 2 * fit_results.log_lik_ratio
  fit_results['p'] = sp.stats.chi2.sf(fit_results.log_lik_ratio_chisq, df=1)
  fit_results['test'] = 'LRT(chisq)'
else:
  fit_results.columns = ['lik', 'lik_cons', 'conv', 'conv_con', 'lambda_lap', 
                        'perm_p', 'perm_conf_nsig', 'p', 'L_rank', 
                        'B_lambda', 'replicate', 'fit_replicate']
  fit_results['test'] = 'Ttest(permutation)'
for param, param_val in info.items():
  fit_results[param] = param_val
  fit_results['sim_no'] = args.sim_number + args.sim_offset

print(fit_results)
fit_results.to_csv(output, sep='\t')
