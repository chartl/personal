import scipy as sp
import scipy.stats
import numpy as np
import pandas as pd
import itertools
import os
import sys
from argparse import ArgumentParser
import gzip

# high_tuning = {k: v[:5] for k, v in do_tuning({'concentration': (1, 500, 10), 'dims': (5, 500, 10), 'oversample': (1, 5, 1), 'd': (1e-3, 1, 5), 'verbose': True}).items()}
# low_tuning = {k: v[:5] for k, v in do_tuning({'concentration': (1, 500, 10), 'dims': (500, 5000, 10), 'oversample': (1, 2, 2), 'd': (0, 5, 6), 'cor_p': (-0.05, -0.025, 0.025, 0.05), 'verbose': True}).items()}

## parameters for simulation
N_REPLICATES = 2
SNP_FREQUENCIES = [0.05, 0.1, 0.2]
SNP_EFFECTS = [0., 0., 0.1, 0.2, 0.5, 1.] #[1 * x for x in [0., 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]]  # = 10
SNP_EFFECT_SD = [1e-9]
NUM_COVARS = [1]
COVAR_EFFECT_SIZE = [0.05]
COVAR_EFFECT_SIZE_SD = [1e-3]
MIX_FRACTION = [1., 0.2, 0.5] #[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  # 11
RESID_H2 = [0.2]
N_SAMPLES = [20, 50, 100, 200]
N_GENES = [20, 50, 100, 150, 200, 500]

def get_simulation_params(parfile=None):
  if parfile is None:
    return {
        'replicates': N_REPLICATES,
        'frequency': SNP_FREQUENCIES,
        'effect': SNP_EFFECTS,
        'effect_sd': SNP_EFFECT_SD,
        'num_covars': NUM_COVARS,
        'covar_effect_size': COVAR_EFFECT_SIZE,
        'covar_effect_size_sd': COVAR_EFFECT_SIZE_SD,
        'mix_fraction': MIX_FRACTION,
        'resid_h2': RESID_H2,
        'n_samples': N_SAMPLES,
        'n_genes': N_GENES
      }
  else:
    pardct = dict()
    for key, vals in (x.strip().split() for x in open(parfile)):
      val_list = vals.split(',')
      if key[0] == 'n':
        val_list = [int(x) for x in val_list]
      elif key == 'replicates':
        val_list = int(val_list[0])
      else:
        val_list = [float(x) for x in val_list]
      pardct[key] = val_list
    defaults = get_simulation_params(None)
    for k in defaults:
      if k not in pardct:
        pardct[k] = defaults[k]
    return pardct


TUNED_PARAMS_HIGH = {
  200: (201, 500.9, 200, 0.0), 
  100: (101, 250.95, 100, 0.0), 
  150: (151, 500.9, 150, 0.0),
   50: (51, 250.95, 50, 0.0), 
   20: (21, 250.95, 20, 0.05), 
  500: (501, 1000.8, 500, 0.0),
 1000: (1001, 5000.0, 1000, 0.0)}

TUNED_PARAMS_LOW = {
  200: (201, 0.01, 200, 4.2), 
  100: (101, 0.01, 100, 5.3999999999999995), 
   50: (51, 0.01, 50, 6.6), 
   20: (21, 0.01, 20, 8.4), 
  500: (4101, 0.01, 500, 1.2)}

def corr_stats(X):
    corr_vec = X[np.tri(X.shape[0], X.shape[0], -1) > 0]
    return np.min(corr_vec), np.mean(corr_vec), np.max(corr_vec)


def rW(n, kappa, m):
    """https://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python"""
    dim = m-1
    b = dim / (np.sqrt(4*kappa*kappa + dim*dim) + 2*kappa)
    x = (1-b) / (1+b)
    c = kappa*x + dim*np.log(1-x*x)
    y = []
    for i in range(0,n):
        done = False
        while not done:
            z = sp.stats.beta.rvs(dim/2,dim/2)
            w = (1 - (1+b)*z) / (1 - (1-b)*z)
            u = sp.stats.uniform.rvs()
            if kappa*w + dim*np.log(1-x*w) - c >= np.log(u):
                done = True
        y.append(w)
    return np.array(y)


def rvMF(n,theta):
    """https://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python"""
    dim = len(theta)
    kappa = np.linalg.norm(theta)
    mu = theta / kappa
    result = []
    for sample in range(0,n):
        w = rW(1, kappa, dim)
        v = np.random.randn(dim)
        v = v / np.linalg.norm(v)
        v2 = np.sqrt(1-w**2)*v + w*mu
        result.append(v2/np.linalg.norm(v2))
    return result


def draw_correlation(a, n, d=None):
    """\
    Creates a covariance [correlation!] matrix by drawing `n` variables
    according to a von-Mises Fisher distribution, parameterized by `a` 

    """
    frame = rvMF(n, a)
    covar = np.dot(np.vstack(frame).T, np.vstack(frame))
    if d is not None:
        covar[np.arange(covar.shape[0]), np.arange(covar.shape[0])] += d
    return covar/np.dot(np.sqrt(np.diag(covar))[:,np.newaxis], np.sqrt(np.diag(covar))[:,np.newaxis].T)


def correlmat(size, conc, ndim, diag):
    a = np.ones(ndim, dtype=np.float64)
    a = a/np.linalg.norm(a)
    return draw_correlation(a*conc, size, diag)


def lrange(a, b, c, dtype=None):
    d = float(b-a)/c
    if dtype is None:
        return np.arange(a, b, d)
    return np.array(np.arange(a, b, d), dtype=dtype)


def tune_covar_params(n_gene, vectors=None, concentration=(1,5000, 20), cor_p=(-0.05, 0.6, 0.9, 0.95), d=(0, 0.05, 3), verbose=False):
    if vectors is None:
        vectors = (n_gene + 1, 10 * n_gene + 1, 5, np.int32)
    for conc in lrange(*concentration):
        for diag in lrange(*d):
            for n_vectors in lrange(*vectors):
                cmat = correlmat(n_vectors, conc, n_gene, diag)
                assert cmat.shape[0] == n_gene, 'bad dim: ' + str(cmat.shape)
                stats = corr_stats(cmat)
                if verbose:
                    print([n_vectors, conc, diag] + list(stats))
                if stats[0] > cor_p[0] and stats[1] > cor_p[1] and stats[1] < cor_p[2] and stats[2] < cor_p[3]: 
                    return (n_vectors, conc, n_gene, diag) 
    return None

def do_tuning(kwargs=None):
    params = dict()
    kwargs = kwargs or dict()
    for n_gene in (20, 50, 100, 250, 500, 1000):
        params[n_gene] = tune_covar_params(n_gene, **kwargs)
        print(params)
    return params


def snp_effects(k, alpha, beta, se=1e-9):
    """\
    Draw `k` SNP effects from a mixture distribution of

    b ~ (1-alpha) * N(0, s) + alpha * N(beta, s)

    but we do not have randomness over `alpha`, exactly k*(1-alpha) and k*alpha
    are drawn from each distribution

    """
    n_null = max(int(k * (1. - alpha)), 0)
    n_alt = k - n_null
    return np.hstack([np.random.normal(0, se, size=(n_null,)), np.random.normal(beta, se, size=(n_alt,))])[np.random.permutation(np.arange(k))]


def snp_dosages(f, n):
    return np.array(np.random.binomial(2, f, size=(n,)), dtype=np.float64)


def covariates_and_effects(n_cov, n_sam, n_gene, eff_mean=0.2, eff_sd=0.2):
    covar = np.random.normal(size=(n_cov, n_sam))
    effects = np.random.normal(eff_mean, eff_sd, size=(n_gene, n_cov))
    return covar, effects


def simulate_data(freq=0.259, n=50, g=100, c=4, alpha=0.2, beta=0.2, se=1e-9, c_mu=0.01, c_sd=0.01, h2=0.2):
    """\
    Simulation a realization of the matrix normal model for a single variant and multiple genes

    :input freq:   the variant frequency. The default is set to the mean frequency of TF Cis-QTLs observed in GTEx
    :input n:      the number of samples
    :input g:      the number of genes
    :input c:      the number of covariates
    :input alpha:  the proportion of genes for which the SNP has an effect
    :input beta:   the (mean) effect size of the SNP
    :input se:     the std-dev of the effect size of the SNP
    :input c_mu:   the (mean) covariate effect size
    :input c_sd:   the std-dev of the covariate effect size
    :input h2:     the heritability of the *residuals*

    """
    dosages = snp_dosages(freq, n)  # n x 1
    betas = snp_effects(g, alpha, beta, se)  # g x 1
    covs, effs = covariates_and_effects(c, n, g, c_mu, c_sd)  # c x n, g x c
    M = np.dot(betas[:,np.newaxis], dosages[:,np.newaxis].T)  # g x n
    F = np.dot(effs, covs)  # g x n
    R = np.random.normal(size=(g, n)) # g x n
    E = correlmat(*TUNED_PARAMS_HIGH[g])  # g x g
    L = np.linalg.cholesky(E)
    G = correlmat(*TUNED_PARAMS_LOW[n]) # n x n
    I = np.diag(np.ones(G.shape[0]))
    W = h2 * G + (1 - h2) * I  # n x n
    H = np.linalg.cholesky(W)
    Y = M + F + np.dot(L, np.dot(R, H.T)) #g x n
    signal = np.mean(np.abs(M))
    noise = np.std(np.dot(L, np.dot(R, H.T)))
    return {'Y': Y, 'x': dosages, 'C': covs, 'D': effs, 'B': betas, 'G': G, 'signal': signal, 'noise': noise, 'snr': signal/noise}


def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def to_gz(df, gz):
    out = gzip.open(gz, 'wb')
    df.to_csv(out, sep='\t')
    out.close()


def run_single_sim(sim_no, param_file=None):
  """\
  Non-main() entry point for simulating data. Instead of iterating over all parameters and
  writing simulation results to disk; enumerate the parameters and simulate data
  for the `sim_no`th parameter set, and return these data, together with the
  parameter settings.

  """
  PARAMS = get_simulation_params(param_file)
  PARAMS = [PARAMS[x][::-1] for x in ['frequency', 'effect', 'effect_sd', 'num_covars',
             'covar_effect_size', 'covar_effect_size_sd', 'mix_fraction', 'resid_h2',
             'n_samples', 'n_genes']]
  param_names = ['frequency', 'effect_size', 'effect_sd', 'num_cov',
                 'cov_effect', 'cov_eff_sd', 'mix_fraction',
                 'residual_h2', 'n_samples', 'n_genes']
  pp = list(itertools.product(*PARAMS))
  if sim_no > len(pp):
    sim_no = sim_no % len(pp) 
  print('simulating: {}/{}'.format(sim_no, len(pp)))
  for i, pn in enumerate(param_names):
    print('  {}: {}'.format(pn, pp[sim_no][i]))
  f, b, bs, c, cm, cs, a, h, n, g = pp[sim_no]
  sim_data = simulate_data(f, n, g, c, a, b, bs, cm, cs, h)
  sim_str = '_'.join(map(str, [f, b, bs, c, cm, cs, a, h, n, g]))
  sim_info = dict(zip(param_names, [f, b, bs, c, cm, cs, a, h, n, g]))
  sim_info['signal'] = sim_data['signal']
  sim_info['noise'] = sim_data['noise']
  sim_info['snr'] = sim_data['snr']
  sim_info['sim_str'] = sim_str
  sim_data['Y'] = pd.DataFrame(sim_data['Y'])
  sim_data['Y'].index = ['Gene{}'.format(1+i) for i in xrange(sim_data['Y'].shape[0])]
  sim_data['Y'].columns = ['Sample{}'.format(1+i) for i in xrange(sim_data['Y'].shape[1])]
  sim_data['x'] = pd.DataFrame(sim_data['x'][:,np.newaxis].T)
  sim_data['x'].index = ['SNP1']
  sim_data['x'].columns = sim_data['Y'].columns
  sim_data['C'] = pd.DataFrame(sim_data['C'])
  sim_data['C'].index = ['Cov{}'.format(i) for i in xrange(sim_data['C'].shape[0])]
  sim_data['C'].columns = sim_data['Y'].columns
  sim_data['G'] = pd.DataFrame(sim_data['G'])
  sim_data['G'].index, sim_data['G'].columns = sim_data['Y'].columns, sim_data['Y'].columns
  return sim_data, sim_info


def main(args):
    mkdir(args.out_dir)
    PARAMS = get_simulation_params(args.param_file)
    replicates = PARAMS['replicates']
    PARAMS = [PARAMS[x][::-1] for x in ['frequency', 'effect', 'effect_sd', 'num_covars',
               'covar_effect_size', 'covar_effect_size_sd', 'mix_fraction', 'resid_h2',
               'n_samples', 'n_genes']]
    param_names = ['frequency', 'effect_size', 'effect_sd', 'num_cov',
                   'cov_effect', 'cov_eff_sd', 'mix_fraction',
                   'residual_h2', 'n_samples', 'n_genes']
    simulation_no, simulation_dir_no, simulation_in_dir = 1, 1, 1
    simulation_dir = '{}/simulations_{}/'.format(args.out_dir, simulation_dir_no)
    mkdir(simulation_dir)
    pp = list(itertools.product(*PARAMS))
    print(PARAMS)
    print(pp[0])
    for f, b, bs, c, cm, cs, a, h, n, g in itertools.product(*PARAMS):
        if simulation_no > args.max_sim:
            break
        for iter_ in xrange(replicates):
            print('simulation {} ...'.format(simulation_no))
            sim_data = simulate_data(f, n, g, c, a, b, bs, cm, cs, h)
            info_file = '{}/simulation_{}.info.txt'.format(simulation_dir, simulation_no)
            gene_file = '{}/simulation_{}.sim_expr.txt.gz'.format(simulation_dir, simulation_no)
            dosage_file = '{}/simulation_{}.dosage.txt.gz'.format(simulation_dir, simulation_no)
            covar_file = '{}/simulation_{}.covariate.txt.gz'.format(simulation_dir, simulation_no)
            grm_file = '{}/simulation_{}.GRM.txt.gz'.format(simulation_dir, simulation_no)
            sim_info = dict(zip(param_names, [f, b, bs, c, cm, cs, a, h, n, g]))
            sim_info['signal'] = sim_data['signal']
            sim_info['noise'] = sim_data['noise']
            sim_info['snr'] = sim_data['snr']
            with open(info_file, 'w') as info_out:
                for k in sorted(sim_info.keys()):
                    info_out.write('{}\t{}\n'.format(k, str(sim_info[k])))
            gene_df = pd.DataFrame(sim_data['Y'])
            gene_df.index = ['Gene{}'.format(1+i) for i in xrange(gene_df.shape[0])]
            gene_df.columns = ['Sample{}'.format(1+i) for i in xrange(gene_df.shape[1])]
            dos_df = pd.DataFrame(sim_data['x'][:,np.newaxis].T)
            dos_df.index = ['SNP1']
            dos_df.columns = gene_df.columns
            covar_df = pd.DataFrame(sim_data['C'])
            covar_df.index = ['Cov{}'.format(i) for i in xrange(covar_df.shape[0])]
            covar_df.columns = gene_df.columns
            grm_df = pd.DataFrame(sim_data['G'])
            grm_df.index, grm_df.columns = gene_df.columns, gene_df.columns
            to_gz(gene_df, gene_file)
            to_gz(dos_df, dosage_file)
            to_gz(covar_df, covar_file)
            to_gz(grm_df, grm_file)
            simulation_no += 1
            simulation_in_dir += 1
            if simulation_in_dir > 50:
                simulation_in_dir = 1
                simulation_dir_no += 1
                simulation_dir = '{}/simulations_{}/'.format(args.out_dir, simulation_dir_no)
                mkdir(simulation_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('--max_sim', default=10**10, help='Cap simulations', type=int)
    parser.add_argument('--param_file', default=None, help='File listing simulation parameters')
    main(parser.parse_args())




