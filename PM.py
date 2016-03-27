import numpy as np

sample = np.loadtxt('sample_14.txt')
sample.__sizeof__()

import matplotlib.pyplot as plt

plt.figure(1, figsize=(25, 20))
for i in range(1, 11):
    bins = i * 10
    plt.subplot(5, 2, i)
    plt.hist(sample, fc='red', bins=bins)
    plt.title(str(bins) + ' bins')
plt.show()

import scipy.stats as stats

x_grid = np.linspace(sample.min(), sample.max())

plt.figure(2, figsize=(20, 10))
bandwidth = 1
kde = stats.gaussian_kde(sample, bw_method=bandwidth / sample.std(ddof=1))
pdf = kde.evaluate(x_grid)
plt.plot(x_grid, pdf, label='bandwidth={0}'.format(bandwidth), linewidth=3, alpha=0.5)
plt.hist(sample, bins=50, fc='red', normed=True)
plt.legend(loc='upper left')
plt.title('KDE ')
plt.show()


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    from sklearn.neighbors import KernelDensity
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


x_grid = np.linspace(sample.min(), sample.max())

fig = plt.figure(3, figsize=(20, 10))
ax = fig.add_subplot(111)
bandwidths = [0.1, 0.2, 0.3, 1.0]
for bandwidth in bandwidths:
    ax.plot(x_grid, kde_sklearn(sample, x_grid, bandwidth=bandwidth), label='bandwidth={0}'.format(bandwidth),
            linewidth=3, alpha=0.5)
ax.hist(sample, 50, fc='red', alpha=0.3, normed=True)
ax.legend(loc='upper left')
plt.title('KDE bandwidth comparision')
plt.show()

fig = plt.figure(4, figsize=(30, 50))
i = 1
for bandwidth in bandwidths:
    plt.subplot(5, 2, i)
    plt.plot(x_grid, kde_sklearn(sample, x_grid, bandwidth=bandwidth), label='bw={0}'.format(bandwidth), linewidth=3,
             alpha=0.5)
    plt.hist(sample, 50, fc='red', normed=True)
    plt.legend(loc='upper left')
    plt.title('KDE bandwidth comparision')
    i += 1

plt.show()

