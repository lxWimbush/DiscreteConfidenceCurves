from discocurves import Hypergeometric, Poisson, Binomial, NegativeBinomial, ArrestedNegativeBinomial
import numpy as np
import scipy.stats as sps
from scipy.special import beta, digamma, betaln, factorial, binom
from scipy.optimize import root_scalar
from functools import cache
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# HGTesting


@cache
def cacheddist_HG(
        k,
        N,
        n,
        support=None,
        alt_range=None,
        cdist=None,
        precision=0.0001,
        intersections=None):
    return Hypergeometric(
        k,
        N,
        n,
        support=support,
        alt_range=alt_range,
        cdist=cdist,
        precision=precision,
        intersections=intersections)


NN = 10000
N, n = 100, 20
plt.figure(figsize=[6, 3])
aps = []
for K in tqdm(range(N + 1)):
    data = sps.hypergeom(N, K, n).rvs(size=NN)
    aps += [np.sort(np.hstack([1 - cacheddist_HG(kk, N, n).possibility(K)
                    for kk in data]))]
    if np.isnan(aps).any():
        break
    plt.plot(aps[-1], np.linspace(0, 1, NN), 'k', alpha=0.3)
plt.plot(np.max(aps, axis=0), np.linspace(0, 1, NN), 'r')
plt.plot([0, 1], [0, 1], 'k:')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.title(r'$K\in\{{0, \dots, 100\}}, N=100, n=20k$')
plt.xlabel(r'Confidence Levels $1-\alpha$')
plt.ylabel(r'Observed rate of coverage')
plt.tight_layout()
A = cacheddist_HG(N=N, n=n, k=10)
print(A.cut(0.01))
print(A.possibility(A.cut(0.01)[0]), A.possibility(A.cut(0.01)[1]))

# PoissonTesting


@cache
def cacheddist_P(
        k,
        max_k=None,
        support=None,
        alt_range=None,
        cdist=None,
        precision=0.0001,
        intersections=None):
    return Poisson(
        k,
        max_k=max_k,
        support=support,
        alt_range=alt_range,
        cdist=cdist,
        precision=precision,
        intersections=intersections)


# Singh Plot
N = 10000
aps = []
tps = []
for l in tqdm(np.linspace(0, 10, 101)):
    data = sps.poisson(l).rvs(size=N)
    aps += [np.sort(np.hstack([1 - cacheddist_P(d, max(1, d)
                    * 20).possibility(l) for d in data]))]

aps = np.array(aps)
plt.figure(figsize=[6, 3])
for a in aps:
    plt.plot(a, np.linspace(0, 1, N), 'k', alpha=0.3)

plt.plot(np.max(aps, axis=0), np.linspace(0, 1, N), 'r')
plt.plot([0, 1], [0, 1], 'k:')
plt.ylim(0, 1)
plt.xlim(0, 1)

plt.title(
    r'$\lambda\in\{{0, 0.1, \dots, 19.9, 10\}}, N={:1.2e}, \overline{{k}}=\max(20, 20k)$'.format(N))
plt.xlabel(r'Confidence Levels $1-\alpha$')
plt.ylabel(r'Observed rate of coverage')
plt.tight_layout()

A = cacheddist_P(10)
print(A.cut(0.01))
print(A.possibility(A.cut(0.01)[0]), A.possibility(A.cut(0.01)[1]))

# Negative Binomial Testing


@cache
def cacheddist_NB(
        k,
        n,
        max_n=None,
        support=None,
        alt_range=None,
        cdist=None,
        precision=0.0001,
        intersections=None):
    return NegativeBinomial(
        k,
        n,
        max_n,
        support,
        alt_range,
        cdist,
        precision,
        intersections)


N = 10000
k = 4
ms = np.zeros(N)
ps = []
for p in tqdm(np.linspace(0.1, 1, 50)):
    ps += [np.sort(np.hstack([1 - cacheddist_NB(k,
                                                int(d + k),
                                                int(d + k) * 4).possibility(p) for d in sps.nbinom.rvs(p=p,
                                                                                                       n=k,
                                                                                                       size=N)]))]
ms = np.max(ps, axis=0)
plt.figure(figsize=[6, 3])
plt.plot(ms, np.linspace(0, 1, N), 'r', linewidth=3)
# plt.plot(tps, np.linspace(0,1,N), 'k', linewidth = 3)
for p in range(50):
    plt.plot(ps[p], np.linspace(0, 1, N), 'k', alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'Confidence Level $1-\alpha$')
plt.ylabel('Observed Rate of Coverage')
plt.legend(labels=['Minimum Observed Coverage',
           r'$p\in\{0.50, 0.51, \dots, 0.99, 1\}$ Coverage'])
plt.plot([0, 1], [0, 1], 'k:')
plt.title(
    r'$k={:d}, p\in\{{0.50, 0.51, \dots, 0.99, 1\}}, \overline{{n}}=10n, N={:1.1e}$'.format(
        k,
        N))
plt.tight_layout()
A = cacheddist_NB(k=4, n=10, max_n=300)
print(A.cut(0.01))
print(A.possibility(A.cut(0.01)[0]), A.possibility(A.cut(0.01)[1]))

# Binomial testing


@cache
def cacheddist_B(
        k,
        n,
        support=None,
        alt_range=None,
        cdist=None,
        intersections=None,
        precision=0.0001):
    return Binomial(k, n, support, alt_range, cdist, intersections, precision)


N = 10000
n = 10
ps = np.linspace(0, 1, 101)
data = sps.binom(p=ps, n=n).rvs(size=[N, len(ps)]).T
aps = None
plt.figure(figsize=[6, 3])
for i in tqdm(range(len(ps))):
    if aps is None:
        aps = np.sort([1 - cacheddist_B(d, n).possibility(ps[i])
                      for d in data[i]])
        plt.plot(aps, np.linspace(0, 1, N), 'k', alpha=0.3)
    else:
        aps = np.vstack(
            (aps, np.sort([1 - cacheddist_B(d, n).possibility(ps[i]) for d in data[i]])))
        plt.plot(aps[-1], np.linspace(0, 1, N), 'k', alpha=0.3)
plt.plot(np.max(aps, axis=0), np.linspace(0, 1, N), 'r')
plt.plot([0, 1], [0, 1], 'k:')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'Confidence Level $1-\alpha$')
plt.ylabel('Observed Rate of Coverage')
plt.legend(labels=['Minimum Observed Coverage',
           r'$p\in\{0, 0.02, \dots, 0.98, 1\}$ Coverage'])
plt.plot([0, 1], [0, 1], 'k:')
plt.title(
    r'$n={:d}, p\in\{{0.02, 0.04, \dots, 0.98, 1\}}, N={:1.1e}$'.format(n, N))
plt.tight_layout()
A = cacheddist_B(k=5, n=10)
print(A.cut(0.01))
print(A.possibility(A.cut(0.01)[0]), A.possibility(A.cut(0.01)[1]))


@cache
def cacheddist_ANB(
        k,
        n,
        max_k,
        max_n,
        support=None,
        alt_range=None,
        cdist=None,
        precision=0.0001,
        intersections=None):
    return ArrestedNegativeBinomial(
        k,
        n,
        max_k,
        max_n,
        support,
        alt_range,
        cdist,
        precision,
        intersections)


# Arrested Negative Binomial Testing
N = 10000
max_n = 20
k = 4
ps = []
for p in tqdm(np.linspace(0, 1, 101)):
    data = sps.bernoulli(p=p).rvs(size=[N, max_n])
    ps += [[1 - cacheddist_ANB(
        k=min(k, sum(d)),
        n=(
            np.where(np.cumsum(d) == k)[0][0]
            + 1 if sum(d) >= k else max_n
        ),
        max_n=max_n,
        max_k=k).possibility(p) for d in data]]
ps = np.sort(ps, axis=1)
ms = np.max(ps, axis=0)
plt.figure(figsize=[6, 3])
plt.plot(ms, np.linspace(0, 1, N), 'r', linewidth=3)
for p in ps:
    plt.plot(p, np.linspace(0, 1, N), 'k', alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'Confidence Level $1-\alpha$')
plt.ylabel('Observed Rate of Coverage')
plt.legend(labels=['Minimum Observed Coverage',
           r'$p\in\{0.50, 0.51, \dots, 0.99, 1\}$ Coverage'])
plt.plot([0, 1], [0, 1], 'k:')
plt.title(
    r'$k={:d}, $'
    + r'$p\in\{{0.50, 0.51, \dots, 0.99, 1\}},$'
    + r'$\overline{{n}}=10n, N={:1.1e}$'.format(k, N))
plt.tight_layout()
A = cacheddist_ANB(k=k, n=max_n, max_n=max_n, max_k=k)
print(A.cut(0.01))
print(A.possibility(A.cut(0.01)[0]), A.possibility(A.cut(0.01)[1]))
