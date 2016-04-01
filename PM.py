"""
Tasks:

1. Sample estimation of PDF (histogram, kernel density estimation)

2. Theoretical distributions PDFs

3. Distribution parameters estimation (graphical method)

4. Biplots (QQ and PP)

5. Statistical hypothesis checking (Kolmogorov Smirnov test, Pearsons chi-squared test, omega-squared test)


Potential distributions:
Weibull distribution (2 parameters) x
Rician distribution (2 parameters) x
Students distribution (1 parameter) x
Rayleigh distribution (1 parameter) x
T location-scale distribution (3 parameters) ?
Noncentral chi-square distribution (2 parameters) ?
Noncentral F distribution (3 parameters) x
Nakagami distribution (2 parameters) x
Loglogistic distribution (2 parameters) ?
Logistic distribution (2 parameters) x
Beta distribution (2 parameters) x
Generalized extreme values distribution (2 parameters) ?
Gamma distribution (2 parameters) x
Lognormal distribution (2 parameters) x
Normal distribution (2 parameters) x
"""

import numpy as np

# load data
sample = np.loadtxt('sample_14.txt')
sample.__sizeof__()



# plotting histogram
# import matplotlib.pyplot as plt

# plt.figure(1, figsize=(25, 20))
# for i in range(1, 11):
#     bins = i * 10
#     plt.subplot(5, 2, i)
#     plt.hist(sample, fc='red', bins=bins, normed=True)
#     plt.title(str(bins) + ' bins')
# plt.show()




# plotting KDE
# import matplotlib.pyplot as plt
# import scipy.stats as stats
#
# x_grid = np.linspace(sample.min(), sample.max())
#
# plt.figure(2, figsize=(20, 10))
# bandwidth = 1
# kde = stats.gaussian_kde(sample, bw_method=bandwidth / sample.std(ddof=1))
# pdf = kde.evaluate(x_grid)
# plt.plot(x_grid, pdf, label='bandwidth={0}'.format(bandwidth), linewidth=3, alpha=0.5)
# plt.hist(sample, bins=50, fc='red', normed=True)
# plt.legend(loc='upper left')
# plt.title('KDE ')
# plt.show()
#
#
# def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
#     """Kernel Density Estimation with Scikit-learn"""
#     from sklearn.neighbors import KernelDensity
#     kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
#     kde_skl.fit(x[:, np.newaxis])
#     # score_samples() returns the log-likelihood of the samples
#     log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
#     return np.exp(log_pdf)
#
#
# x_grid = np.linspace(sample.min(), sample.max())
#
# fig = plt.figure(3, figsize=(20, 10))
# ax = fig.add_subplot(111)
# bandwidths = [0.1, 0.2, 0.3, 1.0]
# for bandwidth in bandwidths:
#     ax.plot(x_grid, kde_sklearn(sample, x_grid, bandwidth=bandwidth), label='bandwidth={0}'.format(bandwidth), linewidth=3, alpha=0.5)
# ax.hist(sample, 50, fc='red', alpha=0.3, normed=True)
# ax.legend(loc='upper left')
# plt.title('KDE bandwidth comparision')
# plt.show()
#
# fig = plt.figure(4, figsize=(30, 50))
# i = 1
# for bandwidth in bandwidths:
#     plt.subplot(5, 2, i)
#     plt.plot(x_grid, kde_sklearn(sample, x_grid, bandwidth=bandwidth), label='bw={0}'.format(bandwidth), linewidth=3,
#              alpha=0.5)
#     plt.hist(sample, 50, fc='red', normed=True)
#     plt.legend(loc='upper left')
#     plt.title('KDE bandwidth comparision')
#     i += 1
#
# plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')

plt.figure(figsize=(45,10))
sns.distplot(sample,kde=False,norm_hist=True)
sns.kdeplot(sample,bw=2)
plt.show()




# plotting distributions
# import matplotlib.pyplot as plt
# import scipy.stats
#
# plt.figure(5, figsize=(30, 10))
# h = plt.hist(sample, bins=100, color='w', fc='red',alpha=0.3, normed=True)
#
# dist_names = ['beta', 'dweibull', 'exponweib', 'genextreme', 'gamma', 'logistic', 'lognorm', 'nakagami', 'norm', 'rice', 't', 'weibull_min', 'weibull_max', 'chi', 'chi2','f']
#
#
# for dist_name in dist_names:
#     dist = getattr(scipy.stats, dist_name)
#     param = dist.fit(sample)
#     pdf_fitted = dist.pdf(scipy.arange(8096), *param[:-2], loc=param[-2], scale=param[-1])
#     plt.plot(pdf_fitted, label="{} with params {}".format(dist_name, param), linewidth=1.5)
#     plt.xlim(0,47)
#     plt.legend(loc='upper center')
# plt.show()
#
#
# import matplotlib.pyplot as plt
# import scipy.stats
#
#
#
# dist_names = [ 'genextreme', 'lognorm', 'f']
#
#
# for dist_name in dist_names:
#     plt.figure(5, figsize=(30, 10))
#     h = plt.hist(sample, bins=100, color='w', fc='red', alpha=0.3, normed=True)
#
#     dist = getattr(scipy.stats, dist_name)
#     param = dist.fit(sample)
#     pdf_fitted = dist.pdf(scipy.arange(8096), *param[:-2], loc=param[-2], scale=param[-1])
#     plt.plot(pdf_fitted, label="{} with params {}".format(dist_name, param), linewidth=2.0)
#     plt.xlim(0,47)
#     plt.legend(loc='upper center')
#     plt.show()
