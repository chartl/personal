"""\

Farm-friendly multiprocessing version of cp_lrt (stupidly parallel)

Instead of providing --var_start and --var_end in the dosages file, instead provide
 1) A list of sites (ids) to process
 2) A central ".do" file for registering the IDs to process
 3) Number of sites to (attempt) to process at a time
 4) A process ID

 This script will
   1) Read the dosages
   2) Poll the ".do" file
   3) Exclude processed IDs
   4) Choose `N` to process
   5) Register these ids
   6) Process these sites
   7) Write output
  + Repeat 2-7 +

"""

from cp_lrt_utils import *
import cp_lrt
import os
import time


def lock(fname):
    lockfile = fname + '.lock'
    while os.path.exists(lockfile):
        time.sleep(1. + np.random.random())
    os.system('touch ' + lockfile)


def unlock(fname):
    lockfile = fname + '.lock'
    os.system('rm ' + lockfile)


def main(args):
    Y, X, C, G, fit_args, reg_args = parse_args(args)
    if reg_args is None and not args.no_regularization:
      reg_args = DEFAULT_REGULARIZATION
    #Y, X, C = scale_matrices(Y, X, C)
    Xi, Xc = X.index, X.columns
    rsids = Xi
    Y, X, C, G = Y.values, X.values, C.values, G.values
    Gsvd = sp.sparse.linalg.svds(G, k=4, which='LM', return_singular_vectors=True)[0]
    Cperm = np.hstack([C.T, Gsvd]).T
    sites_list = [x.strip() for x in open(args.sites_list)]
    for loop_ in xrange(args.n_loops):
        lock(args.do_file)  # will wait until a lock is obtained
        if os.path.exists(args.do_file):
            done_sites = [x.strip() for x in open(args.do_file)]
        else:
            done_sites = []
        remaining_sites = [x for x in sites_list if x not in done_sites]
        process_sites = remaining_sites[:args.n_sites]
        done_sites += process_sites
        with open(args.do_file, 'w') as out:
            out.write('\n'.join(done_sites))
        unlock(args.do_file)
        print('running ' + ','.join(process_sites))
        site_index = [i for i, s in enumerate(rsids) if s in process_sites]
        if len(site_index) == 0:
            break
        X_prime = X[np.array(site_index), :]
        rsids_prime = [rsids[i] for i in site_index]
        fit_results = list()
        for j, rid in enumerate(rsids_prime):
            model_res = cp_lrt.fit_single_variant_restart(Y, X_prime[(j,), :], C, G, args.lra_rank, None, reg_args, fit_args, return_all_params=False, nr=2)[:5]
            fit_results.append(list(model_res) + [np.nan, np.nan])
        fit_results = pd.DataFrame(fit_results)
        fit_results.columns = ['lik', 'lik_cons', 'conv', 'conv_con', 'lambda_lap', 'perm_p', 'perm_conf_nsig']
        fit_results['log_lik_ratio'] = fit_results.lik - fit_results.lik_cons
        fit_results['log_lik_ratio_chisq'] = 2 * fit_results.log_lik_ratio
        fit_results['p_chisq'] = sp.stats.chi2.sf(fit_results.log_lik_ratio_chisq, df=1)
        fit_results.index = rsids_prime
        output_file = args.output + '_{}_{}.txt'.format(args.process_id, args.loop)
        fit_results.to_csv(args.output, sep='\t', na_rep='NA')


if __name__ == '__main__':
    parser = mkparser()
    parser.add_argument('--do_file', help='Central file for registering IDs', default=None)
    parser.add_argument('--process_id', help='The process ID', default=None)
    parser.add_argument('--sites_list', help='List of sites to process', default=None)
    parser.add_argument('--n_sites', help='Number of sites per loop', default=5, type=int)
    parser.add_argument('--n_loops', help='Number of loops to run', default=10, type=int)
    args = parser.parse_args()
    if args.do_file is None:
        raise ValueError('Must provide a `.do` file')
    if args.process_id is None:
        raise ValueError('Must provide a (unique) process ID')
    if args.sites_list is None:
        raise ValueError('Must provide a sites list')
    print(args)
    main(args)
