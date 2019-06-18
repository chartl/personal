"""\
Tester for the combined-phenotype likelihood ratio test

"""
from cp_lrt import *
N_SAMPLE = 150
TESTITER=500
MAXITER=10000
SEED=28
CONV_LIMIT = 1e-5
CONV_LIMIT2 = 1e-2
def vine(k, eta, nu, state):
	beta = eta + (k-1)/2.
	partials = np.zeros((k,k), dtype=np.float32)
	S = np.diag(np.ones(k, dtype=np.float32))
	for d in range(k-1):
		beta -= 0.5
		for j in range(d+1,k):
			partials[d,j] = 2 * (state.beta(nu * beta, beta) - 0.5)
			p = partials[d,j]
			for l in range(d)[::-1]:
				p = p * np.sqrt((1-partials[l,j]**2) * (1-partials[l,d]**2)) + partials[l,j]*partials[l,d]
			S[d, j], S[j, d] = p, p
	idx = state.permutation(range(k))
	S = S[idx, :]
	S = S[:, idx]
	return S

def interactive_test():
	MOM = 0.2
	LR = 0.001
	gen_ = np.random.RandomState(SEED)

	# simulate a small null model
	B = np.array([0.1, -0.1, 0, 0.05, -0.05, 0, 0, 0]).reshape((8,1))
	D = gen_.standard_normal((8, 3))
	C = gen_.standard_normal((3, N_SAMPLE))
	x = gen_.binomial(2, 0.1, size=(1,N_SAMPLE)).astype(np.float32)
	x = (x - 0.2)/np.sqrt(0.1 * 0.9 * 2)
	E = vine(8, 40, 3, gen_)
	L = np.linalg.cholesky(E)
	del E
	s = 2.
	s_e = np.log(s)
	W = s * np.diag(np.ones(N_SAMPLE, dtype=np.float32))
	R = np.dot(np.dot(L, gen_.standard_normal((8, N_SAMPLE))), W)
	Y = np.dot(B, x) + np.dot(D, C) + R
	del s

	simulation_params = (B.copy(), D.copy(), L.copy(), s_e)


	# simulate a couple Nesterov steps manually and log the params
	# do this with one parameter, holding the rest fixed
	Bguess = B + MOM * LR * del_B  # momentum * nesterov; just for the 1st step

	fit_run = list()
	fit_run.append((B.copy(), MOM * LR * del_B))

	del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, D, L, s_e, 0.05)

	B = B + del_B

	for j in xrange(TESTITER):
		fit_run.append((B.copy(), del_B))
		Bguess = B + MOM * del_B 
		del_B = MOM * del_B + LR * deriv_B(Y, x, C, Bguess, D, L, s_e, 0.05)
		B += del_B
		if j % 100 == 0:
			print(likelihood(Y, x, C, B, D, L, s_e, 0.05))

	print(fit_run[-1][1]/LR)

	B = fit_run[0][0]

	Dguess = D + MOM * LR * del_D
	fit_run = list()
	fit_run.append((D.copy(), MOM * LR * del_D))
	del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, B, Dguess, L, s_e, 0.05)

	D += del_D

	for j in xrange(TESTITER):
		fit_run.append((D.copy(), del_D))
		Dguess = D + MOM * del_D
		del_D = MOM * del_D + LR * deriv_D(Y, x, C, B, Dguess, L, s_e, 0.05)
		D += del_D
		if j % 100 == 0:
			print(likelihood(Y, x, C, B, D, L, s_e, 0.05))

	print(fit_run[-1][1]/LR)

	D = fit_run[0][0]

	Lguess = L + MOM * LR * del_L
	fit_run = list()
	fit_run.append((L.copy(), MOM * LR * del_L))
	del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, B, D, Lguess, s_e, 0.05)

	L += del_L

	for j in xrange(TESTITER):
		fit_run.append((L.copy(), del_L))
		Lguess = L + MOM * del_L
		del_L = MOM * del_L + LR * deriv_L(Y, x, C, B, D, Lguess, s_e, 0.05)
		L += del_L
		if j % 100 == 0:
			print(likelihood(Y, x, C, B, D, L, s_e, 0.05))

	print(fit_run[-1][1]/LR)

	L = fit_run[0][0]

	sguess = s_e + MOM * LR * del_s
	fit_run = list()
	fit_run.append((s_e, MOM * LR * del_s))
	del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, B, D, L, sguess, 0.05)

	s_e += del_s

	for j in xrange(TESTITER):
		fit_run.append((s_e, del_s))
		sguess = s_e + MOM * del_s
		del_s = MOM * del_s + LR * deriv_s(Y, x, C, B, D, L, sguess, 0.05)
		s_e += del_s
		if j % 100 == 0:
			print(likelihood(Y, x, C, B, D, L, s_e, 0.05))

	print(fit_run[-1][1]/LR)

	s_e = fit_run[0][0]

	init_params = (B.copy(), D.copy(), L.copy(), s_e)

	## now try all together
	del_B = deriv_B(Y, x, C, B, D, L, s_e, 0.05)
	del_D = deriv_D(Y, x, C, B, D, L, s_e, 0.05)
	del_L = deriv_L(Y, x, C, B, D, L, s_e, 0.05)
	del_s = deriv_s(Y, x, C, B, D, L, s_e, 0.05)

	MOM = 0.5
	LR = 5e-4

	Bguess = B + MOM * LR * del_B
	Dguess = D + MOM * LR * del_D
	Lguess = L + MOM * LR * del_L
	sguess = s_e + MOM * LR * del_s

	del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)

	B += del_B
	D += del_D
	L += del_L
	s_e += del_s


	print('----- Unconstrained 1 -----')
	fit_run = list()
	lp = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	for j in xrange(MAXITER):
		fit_run.append((del_B, del_D, del_L, del_s))
		Bguess = B + MOM * del_B
		Dguess = D + MOM * del_D
		Lguess = L + MOM * del_L
		sguess = s_e + MOM * del_s
		del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		B += del_B
		D += del_D
		L += del_L
		s_e += del_s
		lc = likelihood(Y, x, C, B, D, L, s_e, 0.05)
		if j % 100 == 0:
			print(j, likelihood(Y, x, C, B, D, L, s_e, 0.05), [mean_err(c)/LR for c in (del_B, del_D, del_L)])
		if (lc - lp)/LR < CONV_LIMIT:
			errs = [mean_err(c)/LR for c in (del_B, del_D, del_L)]
			if np.max(errs) < CONV_LIMIT2:
				break
		else:
			lp = lc

	print(fit_run[-1][0]/LR)
	print(fit_run[-1][1]/LR)
	print(fit_run[-1][2]/LR)
	print(fit_run[-1][3]/LR)

	lik = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('initial lik: {}, after fit: {}'.format(init_lik, lik))

	# now test a constrained fit
	B, D, L, s_e = init_params
	B = B - np.mean(B)
	del_B = deriv_B(Y, x, C, B, D, L, s_e, 0.05)
	del_D = deriv_D(Y, x, C, B, D, L, s_e, 0.05)
	del_L = deriv_L(Y, x, C, B, D, L, s_e, 0.05)
	del_s = deriv_s(Y, x, C, B, D, L, s_e, 0.05)

	Bguess = B + MOM * LR * del_B
	Dguess = D + MOM * LR * del_D
	Lguess = L + MOM * LR * del_L
	sguess = s_e + MOM * LR * del_s

	del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)

	del_B = del_B - np.mean(del_B)

	B += del_B
	D += del_D
	L += del_L
	s_e += del_s

	fit_run = list()

	print('----- Constrained 1 -----')
	lp = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	for j in xrange(MAXITER):
		fit_run.append((del_B, del_D, del_L, del_s))
		Bguess = B + MOM * del_B
		Dguess = D + MOM * del_D
		Lguess = L + MOM * del_L
		sguess = s_e + MOM * del_s
		del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_B = del_B - np.mean(del_B)
		del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		B += del_B
		D += del_D
		L += del_L
		s_e += del_s
		if j % 100 == 0:
			print(j, likelihood(Y, x, C, B, D, L, s_e, 0.05), [mean_err(c)/LR for c in (del_B, del_D, del_L)])
		if (lc - lp)/LR < CONV_LIMIT:
			errs = [mean_err(c)/LR for c in (del_B, del_D, del_L)]
			if np.max(errs) < CONV_LIMIT2:
				break
		else:
			lp = lc

	print(fit_run[-1][0]/LR)
	print(fit_run[-1][1]/LR)
	print(fit_run[-1][2]/LR)
	print(fit_run[-1][3]/LR)

	lik_cons = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('initial lik: {}, after fit: {}'.format(init_lik, lik_cons))
	print(np.sum(B))


	fit_run=list()
	# now remove the constraint
	print('----- Unconstrained 1.1 -----'))
	lp = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	for j in xrange(MAXITER):
		fit_run.append((del_B, del_D, del_L, del_s))
		Bguess = B + MOM * del_B
		Dguess = D + MOM * del_D
		Lguess = L + MOM * del_L
		sguess = s_e + MOM * del_s
		del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		B += del_B
		D += del_D
		L += del_L
		s_e += del_s
		if j % 100 == 0:
			print(j, likelihood(Y, x, C, B, D, L, s_e, 0.05), [mean_err(c)/LR for c in (del_B, del_D, del_L)])
		if (lc - lp)/LR < CONV_LIMIT:
			errs = [mean_err(c)/LR for c in (del_B, del_D, del_L)]
			if np.max(errs) < CONV_LIMIT2:
				break
		else:
			lp = lc

	lik = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('constrained: {}, unconstrained: {}, dif: {}, chisq: {}, p: {}'.format(lik_cons, lik, 
		lik - lik_cons, 2 * (lik - lik_cons), sp.stats.chi2.sf(2*(lik - lik_cons), df=1)))


	# now simulate a difinitively alternate model
	B = np.array([0.5, 0.1, 1.2, 0.5, 0.8, 1.1, 0.7, 0.5]).reshape((8,1))
	_, D, L, s_e = simulation_params
	R = np.dot(np.dot(L, gen_.standard_normal((8, N_SAMPLE))), W)
	Y = np.dot(B, x) + np.dot(D, C) + R

	init_params = (B.copy(), D.copy(), L.copy(), s_e)
	del_B = deriv_B(Y, x, C, B, D, L, s_e, 0.05)
	del_D = deriv_D(Y, x, C, B, D, L, s_e, 0.05)
	del_L = deriv_L(Y, x, C, B, D, L, s_e, 0.05)
	del_s = deriv_s(Y, x, C, B, D, L, s_e, 0.05)


	Bguess = B + MOM * LR * del_B
	Dguess = D + MOM * LR * del_D
	Lguess = L + MOM * LR * del_L
	sguess = s_e + MOM * LR * del_s

	del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)

	B += del_B
	D += del_D
	L += del_L
	s_e += del_s

	fit_run = list()

	print('----- Unconstrained 2 -----')
	lp = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	for j in xrange(MAXITER):
		fit_run.append((del_B, del_D, del_L, del_s))
		Bguess = B + MOM * del_B
		Dguess = D + MOM * del_D
		Lguess = L + MOM * del_L
		sguess = s_e + MOM * del_s
		del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		B += del_B
		D += del_D
		L += del_L
		s_e += del_s
		if j % 100 == 0:
			print(j, likelihood(Y, x, C, B, D, L, s_e, 0.05), [mean_err(c)/LR for c in (del_B, del_D, del_L)])
		if (lc - lp)/LR < CONV_LIMIT:
			errs = [mean_err(c)/LR for c in (del_B, del_D, del_L)]
			if np.max(errs) < CONV_LIMIT2:
				break
		else:
			lp = lc

	print(fit_run[-1][0]/LR)
	print(fit_run[-1][1]/LR)
	print(fit_run[-1][2]/LR)
	print(fit_run[-1][3]/LR)

	lik_cons = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('initial lik: {}, after fit: {}'.format(init_lik, lik_cons))
	print(np.sum(B))

	B, D, L, s_e = init_params
	B = B - np.mean(B)
	del_B = deriv_B(Y, x, C, B, D, L, s_e, 0.05)
	del_D = deriv_D(Y, x, C, B, D, L, s_e, 0.05)
	del_L = deriv_L(Y, x, C, B, D, L, s_e, 0.05)
	del_s = deriv_s(Y, x, C, B, D, L, s_e, 0.05)

	Bguess = B + MOM * LR * del_B
	Dguess = D + MOM * LR * del_D
	Lguess = L + MOM * LR * del_L
	sguess = s_e + MOM * LR * del_s

	del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
	del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)

	del_B = del_B - np.mean(del_B)

	B += del_B
	D += del_D
	L += del_L
	s_e += del_s

	fit_run = list()
	print('----- Constrained 2 -----')
	lp = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	for j in xrange(MAXITER):
		fit_run.append((del_B, del_D, del_L, del_s))
		Bguess = B + MOM * del_B
		Dguess = D + MOM * del_D
		Lguess = L + MOM * del_L
		sguess = s_e + MOM * del_s
		del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_B = del_B - np.mean(del_B)
		del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		B += del_B
		D += del_D
		L += del_L
		s_e += del_s
		if j % 100 == 0:
			print(j, likelihood(Y, x, C, B, D, L, s_e, 0.05), [mean_err(c)/LR for c in (del_B, del_D, del_L)])
		if (lc - lp)/LR < CONV_LIMIT:
			errs = [mean_err(c)/LR for c in (del_B, del_D, del_L)]
			if np.max(errs) < CONV_LIMIT2:
				break
		else:
			lp = lc

	print(fit_run[-1][0]/LR)
	print(fit_run[-1][1]/LR)
	print(fit_run[-1][2]/LR)
	print(fit_run[-1][3]/LR)


	lik_cons = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('initial lik: {}, after fit: {}'.format(init_lik, lik_cons))
	print(np.sum(B))


	fit_run=list()
	# now remove the constraint
	lp = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('----- Unconstrained 2.1 -----')
	for j in xrange(MAXITER):
		fit_run.append((del_B, del_D, del_L, del_s))
		Bguess = B + MOM * del_B
		Dguess = D + MOM * del_D
		Lguess = L + MOM * del_L
		sguess = s_e + MOM * del_s
		del_B = MOM * LR * del_B + LR * deriv_B(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_D = MOM * LR * del_D + LR * deriv_D(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_L = MOM * LR * del_L + LR * deriv_L(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		del_s = MOM * LR * del_s + LR * deriv_s(Y, x, C, Bguess, Dguess, Lguess, sguess, 0.05)
		B += del_B
		D += del_D
		L += del_L
		s_e += del_s
		if j % 100 == 0:
			print(j, likelihood(Y, x, C, B, D, L, s_e, 0.05), [mean_err(c)/LR for c in (del_B, del_D, del_L)])
		if (lc - lp)/LR < CONV_LIMIT:
			errs = [mean_err(c)/LR for c in (del_B, del_D, del_L)]
			if np.max(errs) < CONV_LIMIT2:
				break
		else:
			lp = lc

	lik = likelihood(Y, x, C, B, D, L, s_e, 0.05)
	print('constrained: {}, unconstrained: {}, dif: {}, chisq: {}, p: {}'.format(lik_cons, lik, 
		lik - lik_cons, 2 * (lik - lik_cons), sp.stats.chi2.sf(2*(lik - lik_cons), df=1)))


	## now try the fit single gene directly
	B = np.array([0.5, 0.1, 1.2, 0.5, 0.8, 1.1, 0.7, 0.5]).reshape((8,1))
	_, D, L, s_e = simulation_params
	R = np.dot(np.dot(L, gen_.standard_normal((8, N_SAMPLE))), W)
	Y = np.dot(B, x) + np.dot(D, C) + R

	fit_single_gene(Y, C, x, None, None, 0.05, DEFAULT_FIT_ARGS, verbose=2)

def test_model():
    N_GENES=100
    N_SAMPLES=50
    N_COVARS=1
    N_SNPS = 1
    Lrank=3
    Y = np.random.normal(size=(N_GENES, N_SAMPLES))
    C = np.random.normal(size=(N_COVARS, N_SAMPLES))
    X = np.random.normal(size=(1, N_SAMPLES))
    G = np.random.normal(size=(N_SAMPLES, N_SAMPLES))
    G = np.dot(G, G.T)
    Xc, Cc = tf.constant(X, name='X'), tf.constant(C, name='C')
    k_latent_init = np.zeros(1)  # corresponds to k = 1
    B_init = np.zeros((Y.shape[0], X.shape[0]))
    D_init = np.zeros((Y.shape[0], C.shape[0]))
    L_init = np.random.normal(size=(Y.shape[0], Lrank))
    K_init = np.ones((Y.shape[0],))
    k_latent_init = np.zeros(1)  # corresponds to k = 1
    # instantiate the variables
    Lv, Kv = tf.Variable(initial_value=L_init, name='L'), tf.Variable(K_init, name='K_latent')
    k_lv = tf.Variable(initial_value=k_latent_init, name='k_l')
    Bv = tf.Variable(initial_value=B_init, name='B')
    Dv = tf.Variable(initial_value=D_init, name='D')
    # set up the sample covariance here
    k_obs = tf.sigmoid(k_lv)
    s_obs = 1.0 - k_obs
    Gc = tf.constant(G, name='G')
    E_s = s_obs * tf.eye(num_rows=Gc.shape[-1].value, dtype=tf.float64)
    W = k_obs * Gc + E_s
    Jv = tf.linalg.cholesky(W)
    with tf.Session() as sess:
      sess.run(fetches=tf.global_variables_initializer())
      k_, k, s, E, W = sess.run(fetches=[k_lv, k_obs, s_obs, E_s, W])
      J = sess.run(fetches=Jv)
    
    
    
    # establish the mean transform
    M = tf.matmul(Bv, Xc) + tf.matmul(Dv, Cc)
    dist = MatrixNormalLowRankApprox(loc=M, row_scale_diag=latent_elu(Kv), row_scale_perturb_factor=Lv, col_scale=Jv)
    with tf.Session() as sess:
      sess.run(fetches=tf.global_variables_initializer())
      Y = sess.run(fetches=dist.sample())
    
    mnmm=dist
    model_vars = {'J': Jv, 'L': Lv, 'B': Bv, 'D': Dv, 'k': k_lv, 'K': Kv}
    fit_args = {'nesterov': 1e-2, 'progress_iter': 2000, 'maxiter': 50000, 'param_convergence_tol': 1e-3, 'loss_convergence_tol': 1e-3}
    l_L, l_B, l_D, l_K = 0.1, 0.1, 0.1, 0.1
    Yvar = tf.placeholder(dtype=tf.float64, shape=Y.shape, name='Y')
    I = tf.eye(*model_vars['L'].shape.as_list(), dtype=model_vars['L'].dtype)
    # build the loss minimize (- log prob) - [regularization]
    loglik = mnmm.log_prob(Yvar)
    regcomp = l_L * norm(model_vars['L'] - I) + l_B * norm(model_vars['B']) + l_D * norm(model_vars['D'] + l_K * norm(model_vars['k']))
    #loss = -loglik - regcomp
    loss = -loglik
    giter = tf.Variable(0, trainable=False)
    lr_schedule = tf.train.polynomial_decay(fit_args['nesterov'], giter, fit_args['maxiter'], fit_args['nesterov']/1000, 0.5)
    mvvec = [model_vars[x] for x in 'BDLKk']
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_schedule, epsilon=1e-5)
    # opt_obj = optimizer.minimize(loss=loss) old-style
    grads_and_vars = optimizer.compute_gradients(loss, mvvec) # new hotness
    grad_norms = [tf.norm(grad) for grad, var in grads_and_vars]
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step=giter)
    with tf.Session() as sess:
      sess.run(fetches=tf.global_variables_initializer())
      print(' {} | {} | {} | {} | {} | {} | {} | {} | {}   '.format(sfill('iter', len(str(fit_args['maxiter'])), '>'),
                      sfill('loss', MAX_CHARS, '^'),
                      sfill('delta', MAX_CHARS, '^'),
                      sfill('lik', MAX_CHARS, '^'),
                      sfill('B grad', MAX_CHARS, '^'),
                      sfill('D grad', MAX_CHARS, '^'),
                      sfill('L grad', MAX_CHARS, '^'),
                      sfill('K grad', MAX_CHARS, '^'),
                      sfill('k grad', MAX_CHARS, '^')))
      loss_p, lik_p = sess.run(fetches=[loss, loglik], feed_dict={Yvar.name: Y})
      for iter_ in xrange(fit_args['maxiter']):
        gn, _ = sess.run(fetches=[grad_norms, apply_grads], feed_dict={Yvar.name: Y})
        loss_s, lik_s = sess.run(fetches=[loss, loglik], feed_dict={Yvar.name: Y})
        del_loss = loss_s - loss_p
        if np.abs(del_loss) < fit_args['loss_convergence_tol']:
          if all([x < fit_args['param_convergence_tol'] for x in gn]):
            print('Convergence in {} iterations!'.format(iter_))
            break
        if iter_ % fit_args['progress_iter'] == 0:
          print(' {} | {} | {} | {} | {} | {} | {} | {} | {}  '.format(sfill(iter_, len(str(fit_args['maxiter']))), 
              sfloat(loss_s, MAX_CHARS),
                              sfloat(del_loss, MAX_CHARS),
                              sfloat(lik_s, MAX_CHARS),
                              sfloat(gn[0], MAX_CHARS),
                              sfloat(gn[1], MAX_CHARS),
                              sfloat(gn[2], MAX_CHARS),
                              sfloat(gn[3], MAX_CHARS),
                              sfloat(gn[4], MAX_CHARS)))
        loss_p, lik_p = loss_s, lik_s
      param_values = sess.run(fetches=model_vars)
    
    param_values['lik']=lik_s
    param_values['loss']=loss_s
    
    a_scalar = latent_elu(tf.Variable(np.zeros(1), name='foo'))
    dist2 = MatrixNormalLowRankApprox(loc=M, row_scale_diag=latent_elu(Kv), row_scale_perturb_factor=Lv, col_scale=Jv * a_scalar)
    mnmm = dist2
    model_vars = {'J': Jv, 'L': Lv, 'B': Bv, 'D': Dv, 'k': k_lv, 'K': Kv, 'q': a_scalar}
    loglik = mnmm.log_prob(Yvar)
    regcomp = l_L * norm(model_vars['L'] - I) + l_B * norm(model_vars['B']) + l_D * norm(model_vars['D'] + l_K * norm(model_vars['k']))
    #loss = -loglik - regcomp
    loss = -loglik
    giter = tf.Variable(0, trainable=False)
    lr_schedule = tf.train.polynomial_decay(fit_args['nesterov'], giter, fit_args['maxiter'], fit_args['nesterov']/1000, 0.5)
    mvvec = [model_vars[x] for x in 'BDLKkq']
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_schedule, epsilon=1e-5)
    # opt_obj = optimizer.minimize(loss=loss) old-style
    grads_and_vars = optimizer.compute_gradients(loss, mvvec) # new hotness
    grad_norms = [tf.norm(grad) for grad, var in grads_and_vars]
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step=giter)
    fit_args['progress_iter'] = 100
    
    with tf.Session() as sess:
      sess.run(fetches=tf.global_variables_initializer())
      print(' {} | {} | {} | {} | {} | {} | {} | {} | {} | {}  '.format(sfill('iter', len(str(fit_args['maxiter'])), '>'),
                      sfill('loss', MAX_CHARS, '^'),
                      sfill('delta', MAX_CHARS, '^'),
                      sfill('lik', MAX_CHARS, '^'),
                      sfill('B grad', MAX_CHARS, '^'),
                      sfill('D grad', MAX_CHARS, '^'),
                      sfill('L grad', MAX_CHARS, '^'),
                      sfill('K grad', MAX_CHARS, '^'),
                      sfill('k grad', MAX_CHARS, '^'),
                      sfill('q grad', MAX_CHARS, '^')))
      loss_p, lik_p = sess.run(fetches=[loss, loglik], feed_dict={Yvar.name: Y})
      for iter_ in xrange(fit_args['maxiter']):
        gn, _ = sess.run(fetches=[grad_norms, apply_grads], feed_dict={Yvar.name: Y})
        loss_s, lik_s = sess.run(fetches=[loss, loglik], feed_dict={Yvar.name: Y})
        del_loss = loss_s - loss_p
        if np.abs(del_loss) < fit_args['loss_convergence_tol']:
          if all([x < fit_args['param_convergence_tol'] for x in gn]):
            print('Convergence in {} iterations!'.format(iter_))
            break
        if iter_ % fit_args['progress_iter'] == 0:
          print(' {} | {} | {} | {} | {} | {} | {} | {} | {} | {}  '.format(sfill(iter_, len(str(fit_args['maxiter']))), 
              sfloat(loss_s, MAX_CHARS),
                              sfloat(del_loss, MAX_CHARS),
                              sfloat(lik_s, MAX_CHARS),
                              sfloat(gn[0], MAX_CHARS),
                              sfloat(gn[1], MAX_CHARS),
                              sfloat(gn[2], MAX_CHARS),
                              sfloat(gn[3], MAX_CHARS),
                              sfloat(gn[4], MAX_CHARS),
                              sfloat(gn[5], MAX_CHARS)))
        loss_p, lik_p = loss_s, lik_s
    
