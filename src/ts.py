import ee
import pandas as pd
import numpy as np
from numpy import float32, nan, isfinite, nan_to_num
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian1DKernel
from scipy.signal import savgol_filter
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
import geopandas as gpd
from datetime import datetime
import warnings
from random import randint
from decouple import config
import glob

warnings.filterwarnings("ignore")

# Google Earth Engine Authenticate
# ee.Authenticate()

def gee_multi_credentials(credentials_dir):
    def mpb_get_credentials_path():
        credentials_files = ee.oauth.credentials_files
        credential = credentials_files[randint(0, 5)]
        ee.oauth.current_credentials_idx += 1

        return credential

    ee.oauth.current_credentials_idx = 0
    ee.oauth.credentials_files = glob.glob(credentials_dir + '/*.json')

    ee.oauth.get_credentials_path = mpb_get_credentials_path


"""
WHITTAKER-EILERS SMOOTHER in Python 3 using numpy and scipy
based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", 
        Anal. Chem. 2003, (75), 3631-3636
coded by M. H. V. Werts (CNRS, France)
tested on Anaconda 64-bit (Python 3.6.4, numpy 1.14.0, scipy 1.0.0)
Read the license text at the end of this file before using this software.
Warm thanks go to Simon Bordeyne who pioneered a first (non-sparse) version
of the smoother in Python.
"""


def speyediff(N, d, format='csc'):
    """
    (utility function)
    Construct a d-th order sparse difference matrix based on
    an initial N x N identity matrix

    Final matrix (N-d) x N
    """

    assert not (d < 0), "d must be non negative"
    shape = (N - d, N)
    diagonals = np.zeros(2 * d + 1)
    diagonals[d] = 1.
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d + 1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)
    return spmat


def whittaker_smooth(y, lmbd, d=2):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals
    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors

    ---------

    Arguments :

    y       : vector containing raw data
    lmbd    : parameter for the smoothing algorithm (roughness penalty)
    d       : order of the smoothing

    ---------
    Returns :

    z       : vector of the smoothed data.
    """

    m = len(y)
    E = sparse.eye(m, format='csc')
    D = speyediff(m, d, format='csc')
    coefmat = E + lmbd * D.conj().T.dot(D)
    z = splu(coefmat).solve(y)
    return z


# Copyright M. H. V. Werts, 2017
#
# martinus point werts à ens-rennes point fr
#
# This software is a computer program whose purpose is to smooth noisy data.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

# Este método foi proposto em:
# Schwieder, M.; Leitão, P.; Bustamante, M. M. C.; Ferreira, L. G.; Rabe, A.; Hostert, P. Mapping Brazilian savanna
# vegetation gradients with Landsat time series, International Journal of Applied Earth Observation and
# Geoinformation, Volume 52, 2016, Pages 361-370, ISSN 0303-2434, https://doi.org/10.1016/j.jag.2016.06.019.
# (https://www.sciencedirect.com/science/article/pii/S0303243416301003)

# Essa implementação foi proposta como adaptação para áreas agrícolas, e parametrizada em:
# Bendini, H. N. et al. Detailed agricultural land classification in the Brazilian cerrado based on phenological
# information from dense satellite image time series, International Journal of Applied Earth Observation and
# Geoinformation, Volume 82, 2019, 101872, ISSN 0303-2434, https://doi.org/10.1016/j.jag.2019.05.005.
# (https://www.sciencedirect.com/science/article/pii/S0303243418308961)

def rbf_smoother(x):
    ts = x
    workTempRes = 8
    sigmas = np.array([float(4), float(8), float(24)])
    sigmas = sigmas / workTempRes
    da = isfinite(ts).astype(float32)
    fit = 0.
    weightTotal = 0.
    for i, sigmas in enumerate(sigmas):
        kernel = Gaussian1DKernel(sigmas).array
        weight = convolve(da, kernel, boundary='fill', fill_value=0, normalize_kernel=True)
        filterTimeseries = convolve(ts, kernel=kernel, boundary='fill', fill_value=nan, normalize_kernel=True)
        fit += nan_to_num(filterTimeseries) * weight
        weightTotal += weight
    fit /= weightTotal
    return fit


# Build a Sentinel-2 collection

def ts_s2_collection(lon: float, lat: float, start_date: str, end_date: str) -> pd.DataFrame:
    '''
    Extrai a séries temporal do sentinel 2 por meio do ponto informado
    '''

    # Criação da feature Point no earth engine
    geometry = ee.Geometry.Point(lon, lat)

    def maskS2scl(image):
        scl = image.select('SCL')
        sat = scl.neq(1)
        shadow = scl.neq(3)
        cloud_lo = scl.neq(7)
        cloud_md = scl.neq(8)
        cloud_hi = scl.neq(9)
        cirrus = scl.neq(10)
        snow = scl.neq(11)
        return image.updateMask(sat.eq(1)) \
            .updateMask(shadow.eq(1)) \
            .updateMask(cloud_lo.eq(1)) \
            .updateMask(cloud_md.eq(1)) \
            .updateMask(cloud_hi.eq(1)) \
            .updateMask(cirrus.eq(1)) \
            .updateMask(snow.eq(1))

    def maskS2cdi(image):
        cdi = ee.Algorithms.Sentinel2.CDI(image)
        return image.updateMask(cdi.gt(-0.8)).addBands(cdi)

    bands = ee.List(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'QA60', 'SCL', 'cdi'])
    band_names = ee.List(
        ['BLUE', 'GREEN', 'RED', 'REDEDGE1', 'REDEDGE2', 'REDEDGE3', 'NIR', 'BROADNIR', 'SWIR1', 'SWIR2', 'QA60', 'SCL',
         'CDI'])

    sen = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)) \
        .map(maskS2scl) \
        .map(maskS2cdi) \
        .select(bands, band_names)

    sen = sen.map(lambda image: image.addBands(
        image.expression("2.5*((b('NIR')-b('RED'))/(b('NIR')+6 * b('RED')-7.5*b('BLUE')+1e4))").rename('EVI')))

    # Extração da série temporal do EE para o Python
    l = sen.filterBounds(geometry).getRegion(geometry, 5).getInfo()  # for sentinel
    out = [dict(zip(l[0], values)) for values in l[1:]]

    # Transformaçẽos da série temporal de EVI para DataFrame
    df = pd.DataFrame(out)
    df['id'] = df.id.str[0:8]  # sentinel é [0:8]
    df = df.loc[:, ['id', 'EVI']]
    df = df.groupby(by=['id'], as_index=False).max()
    df['id'] = pd.to_datetime(df['id'])
    df = df.sort_values(by=['id'])
    df = df.rename(columns={'id': 'Date'})

    return df


def resample_data(evi_dataframe: gpd.GeoDataFrame, date_start: str,date_end: str, n_day: int) -> gpd.GeoDataFrame:

    '''
    Função para aplicar a interpolação RBF e realizar uma reamostragem com o espaçemento de n_days na série temporal
    '''

    # Data de ínicio e fim da reamostragem
    date_start = np.datetime64(date_start)
    date_end = np.datetime64(date_end)
    data_evi = evi_dataframe

    # Criação do intervalo de espaçados em 8
    data_range = pd.to_datetime(np.arange(date_start, date_end, np.timedelta64(8,"D"),dtype='datetime64[ns]'))
    evi_nan = np.empty(data_range.shape)
    evi_nan[:] = np.nan

    # Criação do dataframe vazio espaçado em 8
    dic_evi8 = {
        'Date':data_range,
        'EVI':evi_nan
    }
    data_evi8 = pd.DataFrame(dic_evi8)
    
    # Combinação ente o dataframe espaçado e o evi_dataframe (série temporal do EVI)
    df_merge = data_evi8.merge(data_evi, on="Date", how="outer", validate="one_to_many")
    df_merge = df_merge.drop(['EVI_x'],axis=1)
    df_merge = df_merge.sort_values('Date', ignore_index=True)
    df_merge = df_merge.rename(columns={'EVI_y':'EVI'})
    df_merge = df_merge.set_index('Date')

    # Aplicação do interpolador RBF
    rbf_smooth = rbf_smoother(df_merge.loc[:,'EVI'].to_numpy())
    df_merge.loc[:,'EVI'] = rbf_smooth

    # Reamostragem com Cubic Spline em n_day dias
    unsample = df_merge.resample(f'{n_day}D').interpolate(method='cubicspline') #bc_type='natural'

    # Criação do intervalo de espaçados em n_days
    data_range = pd.to_datetime(np.arange(unsample.index[0], unsample.index[-1] + pd.Timedelta(1, unit="d"), np.timedelta64(n_day,"D"),dtype='datetime64[ns]'))
    evi_nan = np.empty(data_range.shape)
    evi_nan[:] = np.nan

    # Criação do dataframe vazio espaçado em n_day
    dic_evi_n = {
        'Date':data_range,
        'EVI':evi_nan
    }
    data_evi_n = pd.DataFrame(dic_evi_n)

    # Geração do dataframe para adequar os valores de EVI bruto com as datas idealmente espaçadas
    for index, row in data_evi_n.iterrows():
        sub_dates = data_evi.Date - row.Date
        sub_dates = np.array([abs(int(data.total_seconds()/(60*60*24))) for data in sub_dates])
        if  np.min(sub_dates) >= n_day:
            data_evi_n.loc[index, 'EVI'] = np.nan
        else:
            data_evi_n.loc[index, 'EVI'] = data_evi.loc[np.argmin(sub_dates), 'EVI']

    # Acrescentando EVI Original no dataframe
    unsample.loc[:, 'EVI_Bruto'] = data_evi_n.loc[:, 'EVI'].to_numpy()

    return unsample


def gerate_graph_EVI_CHIRPS_campaign(lon: float, lat: float, start_date: str, end_date: str, day_spaced: int, smooth: str) -> dict:
    gee_multi_credentials(config('GEE_CREDENCIALS_DIR'))
    ee.Initialize()

    # Extrai serie temporal do Sentinel 2
    tent = 0
    while True:
        try:
            ts_evi = ts_s2_collection(lon, lat, start_date, end_date)
        except Exception as e:
            tent += 1
            if (tent == 20):
                raise Exception(f'Error na conexão com o GEE. Mensagem de erro:{e}')
            continue
        break

    # Aplicação do RBF e resample para espacamento de 8 dias
    resample_evi = resample_data(ts_evi, start_date, end_date, day_spaced)

    # Aplicação de suavizador
    if (smooth == 'whittaker'):
        # Whittaker smoother
        resample_evi.loc[:, 'EVI'] = whittaker_smooth(resample_evi.loc[:, 'EVI'].to_numpy(), 10, d=2)
    elif (smooth == 'savgol'):
        # Savitzky-Golay smoother
        resample_evi.loc[:, 'EVI'] = savgol_filter(resample_evi.loc[:, 'EVI'].to_numpy(), window_length=13, polyorder=5)

    # Aplicando o phenometrics
    # x = stmetrics.phenology.phenometrics(resample_evi.loc[:, 'EVI'].to_numpy(), minimum_up=0.1, min_height=0.05,
    #                                      smooth_fraction=0.05, periods=23, treshold=.9, window=11, iterations=1,
    #                                      show=False)
    # length_season = (round(x['Length']) * day_spaced + 15).astype(np.int16)
    # length_season.index = np.arange(1, length_season.shape[0] + 1)
    # start_season = resample_evi.iloc[round(x['Start']).astype(np.int16), :]
    # end_season = resample_evi.iloc[round(x['End']).astype(np.int16), :]

    # Chirps Image Collection
    chirps_pentad = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    # Criação da feature Point no earth engine
    geometry = ee.Geometry.Point(lon, lat)

    # Extração da série temporal do CHIRPS Pentad do EE para o Python
    tent = 0
    while True:
        try:
            chirps_p = chirps_pentad.filterBounds(geometry).filterDate(start_date, end_date).getRegion(geometry,
                                                                                                       5).getInfo()
        except Exception as e:
            tent += 1
            if (tent == 20):
                raise Exception(f'Error na conexão com o GEE. Mensagem de erro:{e}')
            continue
        break

    out = [dict(zip(chirps_p[0], values)) for values in chirps_p[1:]]

    # Transformaçẽos da série temporal de CHIRPS para DataFrame
    df_chirps = pd.DataFrame(out)
    df_chirps = df_chirps.loc[:, ['id', 'precipitation']]
    df_chirps['id'] = pd.to_datetime(df_chirps['id'])
    df_chirps = df_chirps.sort_values(by=['id'])
    df_chirps = df_chirps.rename(columns={'id': 'Date'})
    df_chirps = df_chirps.set_index('Date')
    df_chirps = df_chirps.resample('8D').sum()

    result = {
        'dates': [date.strftime('%Y-%m-%d') for date in resample_evi.index.tolist()],
        'evi': resample_evi.EVI.values.tolist(),
        'precipitation': df_chirps.precipitation.values.tolist()
    }

    ee.Reset()

    return result


def gerate_graph_EVI_CHIRPS_public(lon: float, lat: float, start_date: str, end_date: str, day_spaced: int, smooth: str) -> dict:
    gee_multi_credentials(config('GEE_CREDENCIALS_DIR'))
    ee.Initialize()

    # Extrai serie temporal do Sentinel 2
    tent = 0
    while True:
        try:
            ts_evi = ts_s2_collection(lon, lat, start_date, end_date)
        except Exception as e:
            tent += 1
            if (tent == 20):
                raise Exception(f'Error na conexão com o GEE. Mensagem de erro:{e}')
            continue
        break

    # Aplicação do RBF e resample para espacamento de 8 dias
    resample_evi = resample_data(ts_evi, start_date, end_date, day_spaced)

    # Aplicação de suavizador
    if (smooth == 'whittaker'):
        # Whittaker smoother
        resample_evi.loc[:, 'EVI'] = whittaker_smooth(resample_evi.loc[:, 'EVI'].to_numpy(), 10, d=2)
    elif (smooth == 'savgol'):
        # Savitzky-Golay smoother
        resample_evi.loc[:, 'EVI'] = savgol_filter(resample_evi.loc[:, 'EVI'].to_numpy(), window_length=13, polyorder=5)

    # Chirps Image Collection
    chirps_pentad = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    # Criação da feature Point no earth engine
    geometry = ee.Geometry.Point(lon, lat)

    # Extração da série temporal do CHIRPS Pentad do EE para o Python
    tent = 0
    while True:
        try:
            chirps_p = chirps_pentad.filterBounds(geometry).filterDate(start_date, end_date).getRegion(geometry,
                                                                                                       5).getInfo()
        except Exception as e:
            tent += 1
            if (tent == 20):
                raise Exception(f'Error na conexão com o GEE. Mensagem de erro:{e}')
            continue
        break

    out = [dict(zip(chirps_p[0], values)) for values in chirps_p[1:]]

    # Transformaçẽos da série temporal de CHIRPS para DataFrame
    df_chirps = pd.DataFrame(out)
    df_chirps = df_chirps.loc[:, ['id', 'precipitation']]
    df_chirps['id'] = pd.to_datetime(df_chirps['id'])
    df_chirps = df_chirps.sort_values(by=['id'])
    df_chirps = df_chirps.rename(columns={'id': 'Date'})
    df_chirps = df_chirps.set_index('Date')
    df_chirps = df_chirps.resample('8D').sum()

    result = {
        'dates': [date.strftime('%Y-%m-%d') for date in resample_evi.index.tolist()],
        'evi': resample_evi.EVI.values.tolist(),
        'evi_orginal': resample_evi.EVI_Bruto.values.tolist(),
        'precipitation': df_chirps.precipitation.values.tolist()
    }

    ee.Reset()

    return result
