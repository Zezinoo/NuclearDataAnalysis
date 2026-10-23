"""Numerical helpers for the MCMC homework; equations are explained in the notebook."""
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from scipy import integrate, optimize, special, stats
from mcmc_sampler import mcmc_sampler


def exp_logprob(rate, edges):
    """Exact log probabilities, including the final [last_edge, infinity) bin."""
    edges = np.asarray(edges, dtype=float)
    return -rate * edges[:-1] + np.log(-np.expm1(-rate * np.diff(edges)))


class SafeSampler(mcmc_sampler):
    """Reuse the supplied chain loop with symmetric proposals and log acceptance."""
    def __init__(self, log_target, start, proposal_cov, seed):
        super().__init__()
        self.log_target = log_target
        self.start = np.atleast_1d(start).astype(float)
        self.chol = np.linalg.cholesky(np.atleast_2d(proposal_cov))
        self.rng = np.random.default_rng(seed)
        self.first_proposition = True
        self.accepted = []

    def make_first_proposition(self):
        self.pars = self.start.copy()
        self.test_pars = self.start.copy()

    def make_normal_proposition(self):
        # Proposals outside support are rejected, never redrawn or clipped.
        self.test_pars = self.pars + self.chol @ self.rng.normal(size=len(self.pars))

    def make_model(self):
        pass  # The target computes exact integrated bin probabilities directly.

    def compute_LL(self):
        return float(self.log_target(self.test_pars))

    def test_acceptation(self, LL, LL2):
        accept = bool(np.isfinite(LL2) and np.log(self.rng.random()) < LL2 - LL)
        self.accepted.append(accept)
        return accept

    def sample(self, draws=12000, burn=3000):
        if not np.isfinite(self.log_target(self.start)):
            raise ValueError("The initial point must have finite posterior density.")
        with redirect_stdout(StringIO()):
            full = np.asarray(super().make_chain(draws + burn, "normal"))
        return full[burn:], float(np.mean(self.accepted[burn:])), full


class mcmc_sampler_exp_model(SafeSampler):
    """Exponential model missing from the supplied session-4 base-class file."""
    def __init__(self, data, edges, start, proposal_sd, seed, alpha=1., beta=.01):
        self.hist, _ = np.histogram(data, edges)
        self.edges = np.asarray(edges)
        self.alpha, self.beta = alpha, beta

        def target(pars):
            rate = pars[0]
            if rate <= 0:
                return -np.inf
            return (self.hist @ exp_logprob(rate, self.edges)
                    + (alpha - 1) * np.log(rate) - beta * rate)
        super().__init__(target, [start], [[proposal_sd**2]], seed)


def grid_posterior(data, edges, grid, alpha=1., beta=.01):
    counts = np.histogram(data, edges)[0]
    lp = np.array([counts @ exp_logprob(r, edges) for r in grid])
    lp += (alpha - 1) * np.log(grid) - beta * grid
    pdf = np.exp(lp - lp.max())
    pdf /= integrate.trapezoid(pdf, grid)
    cdf = integrate.cumulative_trapezoid(pdf, grid, initial=0)
    mean = integrate.trapezoid(grid * pdf, grid)
    sd = np.sqrt(integrate.trapezoid((grid - mean)**2 * pdf, grid))
    quantiles = np.interp([.025, .5, .975], cdf, grid)
    return dict(pdf=pdf, mean=mean, sd=sd, lo=quantiles[0],
                median=quantiles[1], hi=quantiles[2])


def split_chains(x):
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]), axis=0)


def rank_normalize(x):
    ranks = stats.rankdata(x.ravel()).reshape(x.shape)
    return stats.norm.ppf((ranks - .375) / (x.size + .25))


def basic_rhat(x):
    n = x.shape[1]
    W = np.var(x, axis=1, ddof=1).mean()
    B = n * np.var(x.mean(axis=1), ddof=1)
    return np.sqrt(((n - 1) * W / n + B / n) / W)


def ess(x):
    """Multichain ESS with FFT autocovariance and positive, monotone lag pairs."""
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(f * f.conjugate(), n=size, axis=1)[:, :n] / n
    W = np.var(x, axis=1, ddof=1).mean()
    var_plus = (n - 1) / n * W + np.var(x.mean(axis=1), ddof=1)
    if var_plus <= 0:
        return float(m * n)
    rho = 1 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1
    pairs = rho[:2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    bad = np.flatnonzero(pairs <= 0)
    if bad.size:
        pairs = pairs[:bad[0]]
    tau = max(1., -1 + 2 * np.minimum.accumulate(pairs).sum()) if len(pairs) else 1.
    return float(m * n / tau)


def diagnostics(chains):
    """Return R-hat, bulk/tail ESS and raw-mean MCSE for each parameter."""
    rows = []
    for j in range(chains.shape[2]):
        raw = chains[:, :, j]
        split = split_chains(raw)
        ranked = rank_normalize(split)
        folded = rank_normalize(np.abs(split - np.median(split)))
        lo, hi = np.quantile(split, [.05, .95])
        rows.append(dict(rhat=max(basic_rhat(ranked), basic_rhat(folded)),
                         bulk_ess=ess(ranked),
                         tail_ess=min(ess((split <= lo).astype(float)),
                                      ess((split >= hi).astype(float))),
                         mcse=raw.std(ddof=1) / np.sqrt(ess(split))))
    return rows


def acf(x, max_lag=150):
    y = np.asarray(x) - np.mean(x)
    f = np.fft.rfft(y, n=2 * len(y))
    c = np.fft.irfft(f * f.conjugate())[:max_lag + 1]
    return c / c[0]


def decode(z):
    z = np.asarray(z)
    return np.stack((np.exp(z[..., 0]), np.exp(z[..., 1]),
                     special.expit(z[..., 2])), axis=-1)


def mixture_logprob(z, edges):
    slow, fast, w = decode(z)
    return np.logaddexp(np.log(w) + exp_logprob(slow, edges),
                       np.log1p(-w) + exp_logprob(fast, edges))


def mixture_nll(z, counts=None, edges=None, data=None):
    if z[0] >= z[1]:
        return 1e100
    if data is not None:
        slow, fast, w = decode(z)
        lp = np.logaddexp(np.log(w * slow) - slow * data,
                         np.log((1 - w) * fast) - fast * data)
        return -float(lp.sum())
    return -float(counts @ mixture_logprob(z, edges))


def mixture_gradient(z, counts=None, edges=None, data=None):
    """Analytic transformed-coordinate score avoids finite-difference line-search noise."""
    if z[0] >= z[1]:
        return np.zeros(3)
    slow, fast, w = decode(z)
    if data is not None:
        ls, lf = np.log(slow)-slow*data, np.log(fast)-fast*data
        ds, df = 1-slow*data, 1-fast*data
        weights = np.ones_like(data)
    else:
        ls, lf = exp_logprob(slow, edges), exp_logprob(fast, edges)
        def derivative(rate):
            width = rate*np.diff(edges)
            correction = np.zeros_like(width)
            finite = np.isfinite(width)
            v = width[finite]
            correction[finite] = v*np.exp(-v)/(-np.expm1(-v))
            return -rate*edges[:-1]+correction
        ds, df = derivative(slow), derivative(fast)
        weights = counts
    lp = np.logaddexp(np.log(w)+ls, np.log1p(-w)+lf)
    responsibility = np.exp(np.log(w)+ls-lp)
    return -np.array([weights @ (responsibility*ds),
                      weights @ ((1-responsibility)*df),
                      weights @ (responsibility-w)])


def fit_mixture(data=None, edges=None, counts=None, start=None, multistart=True):
    if edges is not None and counts is None:
        counts = np.histogram(data, edges)[0]
    objective = lambda z: mixture_nll(z, counts, edges, data if edges is None else None)
    gradient = lambda z: mixture_gradient(z, counts, edges, data if edges is None else None)
    starts = [np.array(start)] if start is not None else [np.array([np.log(100), np.log(10000), .5])]
    if multistart:
        starts += [np.array([np.log(s), np.log(f), w])
                   for s, f, w in [(50, 3000, -1), (200, 20000, 1), (20, 1000, 0)]]
    bounds = [(np.log(1), np.log(1e6))] * 2 + [(-12, 12)]
    # Nelder-Mead handles the ordered support without a line search crossing its barrier.
    # The analytic score independently verifies that the returned optimum is stationary.
    fits = [optimize.minimize(objective, z, method="Nelder-Mead", bounds=bounds,
                             options=dict(xatol=1e-9, fatol=1e-9, maxiter=4000))
            for z in starts]
    fit = min(fits, key=lambda r: r.fun)
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Mixture optimizer failed: {fit.message}")
    if np.linalg.norm(gradient(fit.x), ord=np.inf) > .01:
        raise RuntimeError("Mixture optimizer stopped before reaching a stationary point.")
    return fit


def hessian(fun, x, step=2e-4):
    x = np.asarray(x)
    h = np.zeros((len(x), len(x)))
    e = np.eye(len(x)) * step
    f0 = fun(x)
    for i in range(len(x)):
        h[i, i] = (fun(x + e[i]) - 2 * f0 + fun(x - e[i])) / step**2
        for j in range(i):
            h[i, j] = h[j, i] = (fun(x + e[i] + e[j]) - fun(x + e[i] - e[j])
                                 - fun(x - e[i] + e[j]) + fun(x - e[i] - e[j])) / (4 * step**2)
    return h


def mixture_logposterior(z, counts, edges):
    if not (0 < z[0] < z[1] < np.log(1e6)):
        return -np.inf
    # Log-uniform ordered rates and uniform w. Rate Jacobians cancel 1/(slow*fast).
    # The logit Jacobian remains: log[w(1-w)].
    return (-mixture_nll(z, counts, edges)
            - np.logaddexp(0, -z[2]) - np.logaddexp(0, z[2]))


def deviance(counts, probabilities):
    mu = counts.sum() * probabilities
    positive = counts > 0
    return float(2 * np.sum(counts[positive] * np.log(counts[positive] / mu[positive])))


def show_table(rows, headers, formats=None):
    """Display a portable HTML table without pandas."""
    from html import escape
    from IPython.display import HTML, display
    formats = formats or [None] * len(headers)
    out = ['<table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers)
           + '</tr></thead><tbody>']
    for row in rows:
        cells = [format(v, f) if f else str(v) for v, f in zip(row, formats)]
        out.append('<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in cells) + '</tr>')
    display(HTML(''.join(out) + '</tbody></table>'))
