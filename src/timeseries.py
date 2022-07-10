import ee
import pandas as pd
import numpy as np
import glob
from random import randint
from .sentinel import ts_s2_collection
from .resample import resample_data
from scipy.signal import savgol_filter
from .whittaker import whittaker_smooth
import stmetrics
import warnings
from decouple import config

warnings.filterwarnings("ignore")


def gee_multi_credentials(credentials_dir):
    def mpb_get_credentials_path():
        credentials_files = ee.oauth.credentials_files
        credential = credentials_files[randint(0, 5)]
        ee.oauth.current_credentials_idx += 1

        return credential

    ee.oauth.current_credentials_idx = 0
    ee.oauth.credentials_files = glob.glob(credentials_dir + '/*.json')

    ee.oauth.get_credentials_path = mpb_get_credentials_path


def get_timeseries(lon: float, lat: float, start_date: str, end_date: str, day_spaced: int, smooth: str) -> list:
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
    if smooth == 'whittaker':
        # Whittaker smoother
        resample_evi.loc[:, 'EVI'] = whittaker_smooth(resample_evi.loc[:, 'EVI'].to_numpy(), 10, d=2)
    elif smooth == 'savgol':
        # Savitzky-Golay smoother
        resample_evi.loc[:, 'EVI'] = savgol_filter(resample_evi.loc[:, 'EVI'].to_numpy(), window_length=13, polyorder=5)

    # import web_pdb
    # web_pdb.set_trace()

    # Aplicando o phenometrics
    x = stmetrics.phenology.phenometrics(resample_evi.loc[:, 'EVI'].to_numpy(), minimum_up=0.1, min_height=0.05,
                                         smooth_fraction=0.05, periods=23, treshold=.9, window=11, iterations=1,
                                         show=False)
    length_season = (round(x['Length']) * day_spaced + 15).astype(np.int16)
    length_season.index = np.arange(1, length_season.shape[0] + 1)
    # start_season = resample_evi.iloc[round(x['Start']).astype(np.int16), :]
    # end_season = resample_evi.iloc[round(x['End']).astype(np.int16), :]

    # Chirps Image Collection
    chirps_pentad = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")

    # Criação da feature Point no earth engine
    geometry = ee.Geometry.Point([lon, lat])

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

    array_chirps = df_chirps.precipitation.values
    # Normalização: y = (x - min) / (max - min) para deixá-los na mesma de escala 0 - 1
    # https://machinelearningmastery.com/normalize-standardize-time-series-data-python/
    min = np.min(array_chirps)
    max = np.max(array_chirps)
    chirps_normalized = [float((value - min) / (max - min)) for value in array_chirps.tolist()]

    return {
        "dates": [date.strftime('%Y-%m-%d') for date in resample_evi.index.tolist()],
        "evi": resample_evi.EVI.values.tolist(),
        "precipitation":  chirps_normalized,
    }