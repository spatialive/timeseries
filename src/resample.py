import numpy as np
import pandas as pd
import geopandas as gpd
from .interpolation import rbf_smoother


def resample_data(evi_dataframe: gpd.GeoDataFrame, date_start: str, date_end: str, n_day: int) -> gpd.GeoDataFrame:
    
    '''
    Função para aplicar a interpolação RBF e realizar uma reamostragem com o espaçemento de n_days na série temporal
    '''

    # Data de ínicio e fim da reamostragem
    date_start = np.datetime64(date_start) #'2020-08-01'
    date_end = np.datetime64(date_end) #'2021-10-01'
    data_evi = evi_dataframe

    # Criação do intervalo de espaçados em n_days
    data_range = pd.to_datetime(np.arange(date_start, date_end, np.timedelta64(n_day, "D"), dtype='datetime64[ns]'))
    evi_nan = np.empty(data_range.shape)
    evi_nan[:] = np.nan

    # Criação do dataframe vazio espaçado em n_day
    dic_evi8 = {
        'Date':data_range,
        'EVI':evi_nan
    }
    data_evi8 = pd.DataFrame(dic_evi8)
    
    # Combinação ente o dataframe espaçado e o evi_dataframe (série temporal do EVI)
    df_merge = data_evi8.merge(data_evi, on="Date", how="outer", validate="one_to_many")
    df_merge = df_merge.drop(['EVI_x'], axis=1)
    df_merge = df_merge.sort_values('Date', ignore_index=True)
    df_merge = df_merge.rename(columns={'EVI_y': 'EVI'})
    df_merge = df_merge.set_index('Date')

    # Aplicação do interpolador RBF
    rbf_smooth = rbf_smoother(df_merge.loc[:, 'EVI'].to_numpy())
    df_merge.loc[:, 'EVI'] = rbf_smooth

    # Reamostragem com Cubic Spline em 8 dias
    unsample = df_merge.resample('8D').interpolate(method='cubicspline') #bc_type='natural'
    # unsample = df_merge.resample('8D').max()

    return unsample