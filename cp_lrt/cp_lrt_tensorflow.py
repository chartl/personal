print """Usage note: on Hoffman:

module load python/anaconda2
module load cuda/9.1
module load gcc/7.2.0

conda create -n tensorflow-cpu tensorflow==1.11.0 python=2
source activate tensorflow-cpu
pip install tensorflow-probability==0.4.0

"""

from argparse import ArgumentParser
import scipy as sp
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework import function
from tensorflow.contrib.distributions.python.ops import distribution_util
from affine_matrix_bijection import *

MAX_CHARS=12

"""\
see https://github.com/tensorflow/tensorflow/issues/12071
fix for Nan gradients of tf.norm() around 0
"""
@function.Defun(tf.float64, tf.float64)
def norm_grad(x, dy):
    return dy*(x/(tf.norm(x, ord=2)+1.0e-19))

@function.Defun(tf.float64, grad_func=norm_grad)
def norm(x):
    return tf.norm(x, ord=2)

def norm_axis1(X):
    return tf.map_fn(lambda x: norm(x), X)


def sfill(x, max_chars=10, justify='>'):
    """\
    Fill a string with empty characters
    source: https://github.com/kyleclo/tensorflow-mle/blob/master/util/sprint.py
    """
    return '{}' \
        .format('{:' + justify + str(max_chars) + '}') \
        .format(x)


def sfloat(x, num_chars=10):
    """Stringify a float to have exactly some number of characters\
    source: https://github.com/kyleclo/tensorflow-mle/blob/master/util/sprint.py
    """
    x = float(x)
    num_chars = int(num_chars)
    start, end = str(x).split('.')
    start_chars = len(str(float(start)))
    if start_chars > num_chars:
        #raise Exception('Try num_chars = {}'.format(start_chars))
        return '...'
    return '{}' \
        .format('{:' + str(num_chars) + '.' +
                str(num_chars - start_chars + 1) + 'f}') \
        .format(x)


def build_bijector_lowrank_approx(M, L, K, J):
    """\
    This builds the bijector for transforming a matrix of i.i.d. standard normals
    into a matrix normal with the following parameters

    mean = M
    row_covar = LL.T + K
    col_covar = JJ.T

    """
    KdiagOP = distribution_util.make_diag_scale(loc=None, scale_diag=K)
    L_op = tf.linalg.LinearOperatorLowRankUpdate(base_operator=KdiagOP, u=L, diag_update=None, 
        is_non_singular=True, is_self_adjoint=True, 
        is_positive_definite=True, is_square=True)
    J_op = tf.linalg.LinearOperatorLowerTriangular(J)
    L_bi = AffineLinearOperatorMatrix(shift=None, scale=L_op)
    J_bi = AffineLinearOperatorMatrix(shift=tf.transpose(M), scale=J_op)
    T_ = lambda: tfp.bijectors.Transpose(rightmost_transposed_ndims=2)
    return tfp.bijectors.Chain([T_(), J_bi, T_(), L_bi]), {'K': KdiagOP.to_dense(), 'L': L_op.to_dense(), 'J': J_op.to_dense()}


class ShapedNormal(tfp.python.distributions.TransformedDistribution):
    """\
    Distributions of i.i.d. normals into arbitrary shapes

    """
    def __init__(self, base_shape, validate_args=False):
        # we're only given a base shape like (16, 30)
        raw_shape = np.prod(base_shape)
        super(ShapedNormal, self).__init__(
            distribution=tfp.distributions.MultivariateNormalDiag(
                loc=None,
                scale_diag = tf.ones(raw_shape, dtype=tf.float64)
                ),
            bijector=tfp.bijectors.Reshape(event_shape_in=[raw_shape], event_shape_out=base_shape, name='SNReshape'),
            batch_shape=None,
            event_shape=None,
            validate_args=validate_args,
            name='ShapedNormal'
            )


class MatrixNormalLowRankApprox(tfp.python.distributions.TransformedDistribution):
    """\
    A matrix normal mixed model can be implemented as a transformed distribution. Namely
    a base response of

    MN(0, I_(n, n), I_(m, m)) is a matrix of n*m i.i.d. N(0, 1) random variables.

    Y = np.random.normal((n, m))

    then

    Y' = MN(M, E, G) can, with E = LL.T + K and JJ.T = G be given as

    Y' = M + LYJ.T

    This specific implementation takes L (the row covariance seed) to be fully parameterized, and
    J (the column covariance seed) to be specified up to a constant, with M parameterized by
    parameters of interest X, and nuisance covariates C, so that

    Y' ~ MN(BX + DC, E, k * G); E = LL.T + K [low rank], G = JJ.T [constant]

    Here, L is a (n x r) with r << n matrix, and K is diagonal (n x n). This gives a diagonal-plus-lowrank
    approximation to the row covariance. 

    -~-~- A NOTE ON K -~-~-

    while K represents a diagonal matrix, the design of the API is that we store it only
    as the vector of values, i.e. as a vector. 

    """
    def __init__(self, loc=None,
                 row_scale_diag=None,
                 row_scale_perturb_factor=None, 
                 col_scale=None, 
                 validate_args=False, 
                 allow_nan_stats=False, 
                 name='MatrixNormalLowRankApprox'):
        if loc is None:
            raise ValueError('Missing `loc` parameter')
        if row_scale_diag is None:
            raise ValueError('Missing `row_scale_diag` parameter')
        if row_scale_perturb_factor is None:
            raise ValueError('Missing `row_scale_perturb_factor` parameter')
        if col_scale is None:
            raise ValueError('Misssing `col_scale` parameter')
        bijector, ops = build_bijector_lowrank_approx(loc, row_scale_perturb_factor, row_scale_diag, col_scale)
        self.ops = ops
        super(MatrixNormalLowRankApprox, self).__init__(
            distribution=ShapedNormal(loc.shape),
            bijector=bijector,
            batch_shape=None,
            event_shape=None,
            validate_args=validate_args,
            name=name
            )


def latent_elu(x):
    return tf.where(x >= 0.0, 1 + x, tf.exp(x))


def append_neg_sum(x):
    return tf.concat([x, -tf.reduce_sum(x, 0, keepdims=True)], axis=0)


def build_model_singlegene(y, x, C, G):
    """\
    Build a single-gene mixed model:

    y ~ N(bx + DC, kG + sI)

    (1 x ns) ~ (1 x 1) (1 x ns) + (1 x c) (c x ns) || (ns x ns) + (ns + ns) 

    """
    assert x.shape == y.shape
    assert x.shape[0] == 1
    Xc, Cc, Gc = tf.constant(x, name='x'), tf.constant(C, name='c'), tf.constant(G, name='G')
    Bi, Di = np.random.normal(size=(1,)), np.random.normal(size=(1, C.shape[0]))
    kli, pli = np.zeros(1), np.zeros(1)
    Bv, Dv = tf.Variable(Bi, name='B'), tf.Variable(Di, name='D'), 
    kl, pl = tf.Variable(kli, 'k_lat'), tf.Variable(pli, 'p_lat')
    ko = latent_elu(kl)
    po = tf.sigmoid(pl)
    W = ko * (po * Gc + (1 - po) * tf.eye(num_rows=G.shape[0], dtype=np.float64))
    M = Bv * Xc
    M += tf.matmul(Dv, Cc)
    model = tfp.python.distributions.MultivariateNormalFullCovariance(loc=M, covariance_matrix=W)
    return model, {'B': Bv, 'D': Dv, 'k': kl, 'ko': ko, 'p': pl, 'po': po, 'W': W}


def build_model(Y, X, C, Lrank, G, init_vars=None, constrain_B=False):
    """\
    Build the Tensorflow model for the CP-LRT ML test:

       Y ~ MN(BXi + DC, LL.T + K, kG + sI)

    Note that for the matrix normal, the likelihood

    inv(V)t(X - M)inv(S)(X - M)

    is invariant under the re-definition R = tV, S = S/t, which creates
    a degree of freedom in the parameter space which can lead to convergence
    issues. The ideal would be to set det(V) = det(S) = 1, but this leads
    to some nasty constraints; a medium approach is to let the scale factor
    be determined solely by S = LL.T + K; setting k in [0, 1] and s = (1 - k).

    This, while not forcing |kG + sI| = 1, does impose a constraint, alleviating
    the majority of the definitional issue.

    :input Y: the response (gene x sample)
    :input X: the genotype dosages (snp x sample)
    :input C: the sample covariates (cov x sample)
    :input Lrank: the rank of the approximation to the gene-gene covariance
    :input G: the GRM

    """
    # initialize the constants
    Xc, Cc = tf.constant(X, name='X'), tf.constant(C, name='C')
    # obtain variable initialization (for nonzero values)
    D_init = np.linalg.lstsq(C.T, Y.T, rcond=None)[0].T
    B_init = np.random.normal(size=(Y.shape[0], X.shape[0]))
    L_init = np.random.normal(size=(Y.shape[0], Lrank))
    K_init = np.random.normal(size=(Y.shape[0],))
    if init_vars:
        B_init = init_vars.get('B', B_init)
        D_init = init_vars.get('D', D_init)
        K_init = init_vars.get('K', K_init)
    if constrain_B:
        B_init = B_init - np.mean(B_init)
    k_latent_init = np.zeros(1) # corresponds to k = 0.5
    # instantiate the variables
    Lv, Kv = tf.Variable(initial_value=L_init, name='L'), tf.Variable(initial_value=K_init, name='K_latent')
    k_lv = tf.Variable(initial_value=k_latent_init, name='k_l')
    # establish the mean transform
    Dv = tf.Variable(initial_value=D_init, name='D')
    if constrain_B:
        # parameterization of beta values that sum to 0
      Bv = tf.Variable(initial_value=B_init[:-1,:], name='B')
      Bo = append_neg_sum(Bv)
      M = tf.matmul(Bo, Xc) + tf.matmul(Dv, Cc)
      B_sum = tf.reduce_sum(Bo)
    else:
      Bv = tf.Variable(initial_value=B_init, name='B')
      M = tf.matmul(Bv, Xc) + tf.matmul(Dv, Cc)
      B_sum = tf.reduce_sum(Bv)
    # set up the sample covariance here
    k_obs = tf.sigmoid(k_lv)
    s_obs = np.ones(1) - k_obs
    if G is not None:
        Gc = tf.constant(G, name='G')
        W = k_obs * Gc + s_obs * tf.eye(num_rows=G.shape[-1], dtype=np.float64)
        Jv = tf.linalg.cholesky(W)
    else:
        Jv = s_obs * tf.eye(num_rows=C.shape[-1], dtype=np.float64)
    dist = MatrixNormalLowRankApprox(loc=M, row_scale_diag=latent_elu(Kv), row_scale_perturb_factor=Lv, col_scale=Jv)
    # verify that the model actually runs
    Yv = tf.placeholder(dtype=tf.float64, shape=Y.shape, name='Ytest')
    with tf.Session() as sess:
      sess.run(fetches=tf.global_variables_initializer())
      lik = sess.run(fetches=dist.log_prob(Yv), feed_dict={Yv.name: Y})
    assert not np.isnan(lik)
    return dist, {'L': Lv, 'B': Bv, 'D': Dv, 'k': k_obs, 'J': Jv, 
                  'K': Kv, 'k_lat': k_lv, 'k_obs': k_obs, 's_obs': s_obs,
                  'E.det': tf.linalg.det(latent_elu(Kv) * tf.eye(num_rows=Y.shape[0], dtype=np.float64) + tf.matmul(Lv, tf.transpose(Lv))),
                  'W.det': tf.linalg.det(W), 'M': M, 'B_sum': B_sum}


def print_helper(iter_, loss, delta, lik, names, gradients):
    if iter_ == 0:
        fields = ['loss', 'delta', 'lik'] + ['{} grad'.format(n) for n in names]
        filled = [sfill('iter', MAX_CHARS, '>')] + [sfill(x, MAX_CHARS, '^') for x in fields]
        print('   ' + ' | '.join(filled))
    fields = [loss, delta, lik] + gradients
    filled = [sfill(iter_, MAX_CHARS, '>')] + [sfloat(x, MAX_CHARS) for x in fields]
    print('   ' + ' | '.join(filled))


def fit_model(Y, mnmm, model_vars, fit_args, reg_params=None, tf_session=None, verbose=True):
    # initialize necessary variables (placeholder for Y, target regularization on L)
    Yvar = tf.placeholder(dtype=tf.float64, shape=Y.shape, name='Y')
    # build the loss minimize (- log prob) - [regularization]
    loglik = mnmm.log_prob(Yvar)
    loss = -loglik
    if reg_params is not None:
        regcomp = 0
        for varname, (lambda_, target) in reg_params.items():
            if target:
                regcomp = regcomp + lambda_ * norm(model_vars[varname] - target)
            else:
                regcomp = regcomp + lambda_ * norm(model_vars[varname])
        loss += regcomp
    giter = tf.Variable(0, trainable=False)
    mvtrainable = {name: var for name, var in model_vars.iteritems() if 'trainable' in dir(var) and var.trainable}
    mvtnames = mvtrainable.keys()
    mvvec = mvtrainable.values()
    if fit_args['cycleiter'] < fit_args['maxiter']:
      lr_schedule = tf.train.polynomial_decay(fit_args['nesterov'], giter, fit_args['cycleiter'], fit_args['nesterov_final'], fit_args['nesterov_power'], cycle=True)
    else:
      lr_schedule = tf.train.polynomial_decay(fit_args['nesterov'], giter, fit_args['maxiter'], fit_args['nesterov_final'], fit_args['nesterov_power'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_schedule, epsilon=0.1)
    # opt_obj = optimizer.minimize(loss=loss) old-style
    grads_and_vars = optimizer.compute_gradients(loss, mvvec) # new hotness
    grad_norms = [tf.norm(grad) for grad, var in grads_and_vars]
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step=giter)
    tf_session = tf_session or tf.Session()
    with tf_session as sess:
        sess.run(fetches=tf.global_variables_initializer())
        all_vars = sess.run(fetches=model_vars)
        loss_p, lik_p = sess.run(fetches=[loss, loglik], feed_dict={Yvar.name: Y})
        converged = 1
        for iter_ in xrange(fit_args['maxiter']):
            gn, _ = sess.run(fetches=[grad_norms, apply_grads], feed_dict={Yvar.name: Y})
            loss_s, lik_s = sess.run(fetches=[loss, loglik], feed_dict={Yvar.name: Y})
            del_loss = loss_s - loss_p
            if np.abs(del_loss) < fit_args['loss_convergence_tol']:
                if all([x < fit_args['param_convergence_tol'] for x in gn]):
                    print('Convergence in {} iterations!'.format(iter_))
                    break
            if iter_ % fit_args['progress_iter'] == 0 and verbose:
                print_helper(iter_, loss_s, del_loss, lik_s, mvtnames, gn)
                #bnorm = sess.run(fetches=norm(model_vars['B']))
                # print 'B norm: {}'.format(bnorm)
                #bsum = sess.run(fetches=model_vars['B'])
                #print bsum
            loss_p, lik_p = loss_s, lik_s
        else:
            converged = 0
            print('Convergence failed')
        param_values = sess.run(fetches=model_vars)
    param_values['loss'] = loss_s
    param_values['lik'] = lik_s
    param_values['converged'] = converged
    return mnmm, param_values

