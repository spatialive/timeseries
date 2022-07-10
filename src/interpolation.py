import numpy as np
from numpy import float32, nan, isfinite, nan_to_num
from astropy.convolution.kernels import Gaussian1DKernel
from astropy.convolution import convolve

"""
 Este método foi proposto em: 
 Schwieder, M.; Leitão, P.; Bustamante, M. M. C.; Ferreira, L. G.; Rabe, A.; Hostert, P. Mapping Brazilian savanna
 vegetation gradients with Landsat time series, International Journal of Applied Earth Observation and 
 Geoinformation, Volume 52, 2016, Pages 361-370, ISSN 0303-2434, https://doi.org/10.1016/j.jag.2016.06.019. 
 (https://www.sciencedirect.com/science/article/pii/S0303243416301003)

 Essa implementação foi proposta como adaptação para áreas agrícolas, e parametrizada em:
 Bendini, H. N. et al. Detailed agricultural land classification in the Brazilian cerrado based on phenological 
 information from dense satellite image time series, International Journal of Applied Earth Observation and 
 Geoinformation, Volume 82, 2019, 101872, ISSN 0303-2434, https://doi.org/10.1016/j.jag.2019.05.005.
 (https://www.sciencedirect.com/science/article/pii/S0303243418308961)
"""
def rbf_smoother(x):

    ts=x
    workTempRes=8
    sigmas=np.array([float(4),float(8),float(24)])
    sigmas=sigmas / workTempRes
    da=isfinite(ts).astype(float32)
    fit = 0.
    weightTotal = 0.
    for i, sigmas in enumerate(sigmas):
        kernel = Gaussian1DKernel(sigmas).array
        weight = convolve(da, kernel, boundary='fill', fill_value=0, normalize_kernel=True)
        filterTimeseries = convolve(ts, kernel, boundary='fill', fill_value=nan, normalize_kernel=True)
        fit += nan_to_num(filterTimeseries) * weight
        weightTotal += weight
    fit /= weightTotal
    return fit
