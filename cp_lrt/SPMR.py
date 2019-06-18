"""\
A bootstrap test for TWAS-signed peripheral master regulator

"""
import itertools
from cp_lrt_utils import *
import cp_lrt_simulations 

global _log
_log = None
def printlog(x):
    if _log is not None:
        _log.write(repr(x) + '\n')
    print(x)


def mkparser():
    parser = ArgumentParser('SPMR.py')
    parser.add_argument('expression', help='The gene expression matrix to test for consistent effect directions', type=str, default=None)
    parser.add_argument('dosages', help='The genotype dosage matrix', type=str, default=None)
    parser.add_argument('covariates', help='The covariate matrix', type=str, default=None)
    parser.add_argument('TWAS', help='The TWAS file (.dat)')
    parser.add_argument('output', help='The output file', type=str, default=None)
    parser.add_argument('--n_bootstrap', help='The number of individual bootstraps to run', type=int, default=2000)
    parser.add_argument('--individual_ids', help='File containing Individual IDs (for individual replicates), corresponding to expression observations', default=None)
    parser.add_argument('--iid_column', help='The column name (in covariates) listing the individual IDs')
    parser.add_argument('--gene_map', help='A file mapping TWAS gene (first column) to expression gene (second column)')
    parser.add_argument('--gene_set', help='A file containing a list of genes to test (subset of all expression genes)')
    parser.add_argument('--gene_blacklist', help='A file containing a list of genes to blacklist (TFs, DNA-bp, etc)')
    parser.add_argument('--joint_geno_covar', help='Run dosages jointly with covariates instead of regressing covariates first', action='store_true')
    parser.add_argument('--no_log', help='no log file', action='store_true')
    parser.add_argument('--min_Z', help='minimum (absolute) Z-score', type=float, default=None)
    parser.add_argument('--double_bootstrap', help='Boostrap over genes and individuals (can better account for LD)', action='store_true')
    return parser


def get_twas_weights(twas_base, gene_map, min_Z=None, standardize=False):
    data = pd.read_csv(twas_base, sep='\\s+')
    if gene_map is None:
        data['gene'] = data['ID']
    else:
        data['gene'] = [gene_map.get(x, np.nan) for x in data['ID']]
    printlog(data.head())
    weights = dict()
    for g, z in zip(data['gene'], data['TWAS.Z']):
        if np.isfinite(z) and str(g).upper() not in {'NA', 'NAN'}:
            if min_Z is None or abs(z) > min_Z:
                weights[g] = z
    if standardize:
        zvec = weights.values()
        pct = np.argsort(np.argsort(zvec))/len(zvec)
        gap = 1 - pct[-1]
        pct = pct + gap/2.
        zstd = sp.stats.norm.ppf(pct)
        weights = dict(zip(weights.keys(), zstd))
    return weights


def read_gene_map(mapfile):
    if mapfile:
        map_ = dict()
        for line in open(mapfile):
            fields = line.strip().split()
            map_[fields[0]] = fields[1]
    else:
        map_ = None
    return map_


def safe_dos_std(A, axis):
    """\
    High or low-frequency markers can, under a bootstrap,
    have 0 standard deviation. In these cases the minor AF
    is capped at 1e-6, and the variance is computed using
    the binomial formula

    """
    s = np.nanstd(A, axis)
    if any(s < 1e-8):
        m = np.nanmean(A, axis)
        j = np.where(s < 1e-8)[0]
        m[m<1e-6] = 1e-6
        m[m>1-1e-6] = 1-1e-6
        s[j] = np.sqrt(2*m[j]*(1-m[j]))
    return s


def simple_cor(Y, X, scale_Y=True, scale_X=True):
    """\
    Given a (n_g x n_s) matrix Y
          a (n_m x n_s) matrix X

    center and scale the rows, and compute

    np.dot(Y, X.T)

    to generate the (gene x marker) matrix of correlations

    """
    if scale_Y:
        Y = (Y - np.mean(Y, 1)[:, np.newaxis])/(np.std(Y, 1))[:, np.newaxis]
    if scale_X:
        X = (X - np.mean(X, 1)[:, np.newaxis])/(safe_dos_std(X, 1))[:, np.newaxis]
    return np.dot(Y, X.T)


def lmfull_cor(Y, X):
    """\
    Given a (n_g x n_s) matrix Y
          a (n_s x [1+n_c]) matrix X

    compute the dosage coefficient b0 for each gene, 
    and standardize it to the correlation scale

    """
    ystd = np.std(Y, 1) # n_g
    xstd = np.std(X[:, 0]) # scalar
    b0 = np.linalg.lstsq(X, Y.T)[0][0,:] # n_g
    return b0 * xstd/ystd


def bootstrap_gen(id_vec, seed=int(1000*np.pi), maxiter=10**4):
    # generate a set of indexes corresponding to an *individual* bootstrap
    gen = np.random.RandomState(seed)
    indivs = np.unique(id_vec)
    indeces = {id_: np.where(id_vec == id_)[0] for id_ in indivs}
    for _ in range(maxiter):
        boot_ids = gen.choice(indivs, len(indivs), replace=True)
        idx = np.concatenate([indeces[x] for x in boot_ids])
        yield idx

def gene_bootstrap_gen(n_genes, seed=int(1000*np.exp(1)), maxiter=10**4):
	# generate a set of gene indexes for a bootstrap
	gen = np.random.RandomState(seed)
	for _ in range(maxiter):
		yield gen.choice(n_genes, n_genes, replace=True)


def bootstrap_generator(id_vec, n_genes, do_genes, i_s=int(1000*np.pi), g_s=int(1000*np.exp(1)), maxiter=10**4):
	id_gen = bootstrap_gen(id_vec, i_s, maxiter)
	if do_genes:
		g_gen = gene_bootstrap_gen(n_genes, g_s, maxiter)
	else:
		g_gen = itertools.repeat(np.arange(n_genes), maxiter)
	for _ in range(maxiter):
		ix = next(id_gen)
		gx = next(g_gen)
		yield ix, gx


def run_bootstrap(expr_mat, dos_mat, cov_mat, weight_vec, iids, args):
    """\

    Dimensions:
     expr_mat  -- (n_g x n_s)
      dos_mat  -- (n_m x n_s)
      cov_mat  -- (n_s x n_c)
      weights  -- (n_g x 1)
         iids  -- (n_s x 1)

    Given the per-gene model

    yi ~ bi*d + C*ci
      ai - intercept
      bi - dosage coefficient
      ci - covariate coefficients

    and per-gene weights wi, compute the test statistic

    theta = sum_i {wi * bi}

    for each dosage `d`

    It is assumed that an intercept term is present in `C`.

    --

    Statistically, the appropriate model to run is to define D=[d, C] and set

    bi_l = dot(inv(DD.T),dot(D.T,yi))[0]  # un-normalized
    bi = bi_l * sd(d)/sd(yi)  # normalized

    However, the default model both mean and variance biased towards zero by
    first defining

    R = Y - C * inv(CC.T) * C.T Y

    and fitting only the model

    ri ~ bi * d

    in which case

    bi_l = cor(ri, d) * sd(ri)/sd(d)
    bi = cor(ri, d)

    However if `args.joint_geno_covar` is set to True, the full model is run

    """
    theta = np.zeros((dos_mat.shape[0], args.n_bootstrap), dtype=np.float32)
    d_ = np.linalg.norm(weight_vec)
    if args.joint_geno_covar:
        CORR = np.zeros((expr_mat.shape[0], dos_mat.shape[0]), dtype=np.float32)
        for i_b, (bsi, gsi) in enumerate(bootstrap_generator(iids, expr_mat.shape[0], args.double_bootstrap, maxiter=theta.shape[1])):
            # build out CORR column by column
            Yb, Cb, Xb = expr_mat[:, bsi][gsi,:], cov_mat[bsi, :], dos_mat[:, bsi]
            for i_m in range(Xb.shape[0]):
                Db = np.hstack([Xb[i,:], Cb])
                CORR[:, i_m] = lmfull_cor(Yb, Db)
            theta[:, i_b] = np.dot(CORR.T, weight_vec[gsi])/d_
    else:
        printlog('Correcting covariates')
        Q = np.linalg.lstsq(cov_mat, expr_mat.T)[0]
        R = expr_mat - np.dot(cov_mat, Q).T
        for i_b, (bsi, gsi) in enumerate(bootstrap_generator(iids, expr_mat.shape[0], args.double_bootstrap, maxiter=theta.shape[1])):
            printlog('Bootstrap %d/%d' % (i_b, theta.shape[1]))
            Rb, Xb = R[:, bsi][gsi,:], dos_mat[:, bsi]
            CORR = simple_cor(Rb, Xb)
            theta[:, i_b] = np.dot(CORR.T, weight_vec[gsi])/d_
    print('Finalizing...')
    boot_res = {
        'mean': np.mean(theta, axis=1),
        'median': np.median(theta, axis=1),
        'sd': np.std(theta, axis=1),
        'skew': sp.stats.skew(theta, axis=1),
        'kurtosis': sp.stats.kurtosis(theta, axis=1),
        'Z_0': np.mean(theta,axis=1)/np.std(theta,axis=1),
        'P_0_boot_lt': np.mean(theta <= 0, axis=1),
        'P_0_boot_gt': np.mean(theta >= 0, axis=1),
        'QL30': np.percentile(theta, 0.1, axis=1),
        'QL20': np.percentile(theta, 1, axis=1),
        'QL10': np.percentile(theta, 10, axis=1),
        'UL10': np.percentile(theta, 90, axis=1),
        'UL20': np.percentile(theta, 99, axis=1),
        'UL30': np.percentile(theta, 99.9, axis=1)
        }
    boot_res['P_0'] = sp.stats.norm.sf(np.abs(boot_res['Z_0']))*2
    print('Fitting T distributions...')
    params = np.zeros((theta.shape[0],3), dtype=np.float32) - 1
    pvals = np.zeros((theta.shape[0],), dtype=np.float32) - 1
    for i in range(theta.shape[0]):
      if i % 15000 == 0:
        print(' .. %d/%d' % (i, theta.shape[0]))
      if boot_res['P_0'][i] < 1e-3:
        params[i,:] = np.array(sp.stats.t.fit(theta[i,:]))
        pvals[i] = 2*sp.stats.t.cdf(0, loc=np.abs(params[i,1]), scale=params[i,2], df=params[i,0])
      elif np.random.random() < 1e-2:  # there are ~200K points, so 2K fits
        params[i,:] = np.array(sp.stats.t.fit(theta[i,:]))
        pvals[i] = 2*sp.stats.t.cdf(0, loc=np.abs(params[i,1]), scale=params[i,2], df=params[i,0])
    boot_res['T_df'] = params[:,0]
    boot_res['T_loc'] = params[:,1]
    boot_res['T_scale'] = params[:,2]
    boot_res['P_T'] = pvals
    return pd.DataFrame(boot_res)


def main(args):
    Y, X, C, _, fit_args, reg_args = parse_args(args)
    twas_weights = get_twas_weights(args.TWAS, read_gene_map(args.gene_map), min_Z=args.min_Z)
    rsids, sm_dos_ids = X.index, X.columns
    expr_genes, sm_expr_ids = [x for x in Y.index], [x for x in Y.columns]
    if args.iid_column is not None:
        ind_ids = np.array(C.loc[:,args.iid_column].tolist())
        C = C.drop(args.iid_column)
    elif args.individual_ids:
        ind_ids = np.array([x.strip() for x in open(args.individual_ids)])
    else:
        ind_ids = np.array([x for x in sm_dos_ids])
    n_tot = len(expr_genes)
    printlog(expr_genes[:8])
    printlog(list(twas_weights.keys())[:8])
    genes_to_test = [x for x in expr_genes if x in twas_weights]
    n_twa = len(genes_to_test)
    gene_table = {'total': n_tot, 'in_twas': n_twa}
    if args.gene_set:
        gs_ = {x.strip() for x in open(args.gene_set)}
        n_igs = len([x for x in expr_genes if x in gs_])
        genes_to_test = [x for x in genes_to_test if x in gs_]
        n_bot = len(genes_to_test)
        gene_table['in_geneset'] = n_igs
        gene_table['both'] = n_bot
    if args.gene_blacklist:
        gs_ = {x.strip() for x in open(args.gene_blacklist)}
        genes_to_test = [x for x in genes_to_test if x not in gs_]
        gene_table['not_in_blacklist'] = len(genes_to_test) 
    printlog('Gene summary:')
    for k, v in gene_table.items():
        printlog('%s: %d' % (k, v))
    Y = Y.loc[np.array(genes_to_test),:]
    W = np.array([twas_weights[g] for g in genes_to_test])
    Y, X, C = Y.values, X.values, C.values
    theta_df = run_bootstrap(Y, X, C, W, ind_ids, args)
    col_order = ['rsid', 'mean', 'sd', 'Z_0', 'P_0', 'P_0_boot_lt', 
         'P_0_boot_gt', 'QL30', 'QL20', 'QL10', 'median', 
         'UL10', 'UL20', 'UL30', 'skew', 'kurtosis', 
         'T_df', 'T_loc', 'T_scale', 'P_T']
    theta_df['rsid'] = rsids
    theta_df = theta_df.loc[:, col_order]
    theta_df.to_csv(args.output, sep='\t', index=False)


if __name__ == '__main__':
    parser = mkparser()
    args = parser.parse_args()
    if not args.no_log:
        _log = open(args.output + '.log', 'w')
    printlog(args)
    main(args)
