import pandas as pd
from pymongo import MongoClient
import datetime as dt
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from scipy import stats

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json




# Funcion que obtiene los datos de un ticker en un tiempo determinado
def get_data_ticker(ticker:str, tf: str, start_date:dt.datetime , end_date: dt.datetime, columns: list=None) -> pd.DataFrame:
    
    # Datos para la conexion a MongoDB
    path = "mongodb+srv://guancho:Julio%407%408%408@marketdata.nclhx.mongodb.net/test"
    client = MongoClient(path)
    db = client[ticker]

    # Convertimos date.date a datetime.dattime
    start_date = dt.datetime.combine(start_date, dt.time.min)
    end_date = dt.datetime.combine(end_date, dt.time.max)

    # Obtenemos los datos del SP en 5M desde start_date hasta end_date
    document = db[tf].find({'Time': {"$gte": start_date, "$lt": end_date}})
    # Creamos un DataFrame con los datos obtenidos
    df = pd.DataFrame(list(document))

    # Nos quedamos sólo con las columnas que nos interesan
    return df[columns] if columns!=[] else df


# --- Volume Profile ---


def calculate_atr(data, period=14):
    """
    Calcula el Average True Range (ATR) para un DataFrame.
    
    Parámetros:
    data (pd.DataFrame): DataFrame que contiene las columnas 'High', 'Low' y 'Close'.
    period (int): El período sobre el cual calcular el ATR.
    
    Retorna:
    pd.Series: Serie con los valores del ATR.
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr

def normalize_profile(profile, total_volume):  
    return [value / total_volume if total_volume>0 else 0 for value in profile ]  

def interpolate_profile(profile_normalized, num_intervals):  
    return np.interp(
        np.linspace(0, len(profile_normalized), num_intervals),
        np.arange(len(profile_normalized)),
        profile_normalized,
    )  
                       
# Funcion que pretrata el VolumeProfile
def volume_profile(data: pd.DataFrame, num_intervals: int=10) -> pd.DataFrame:
    data['VolumeProfile'] = list(map(lambda ask,bid: [ int(a)+int(b) for a,b in zip(ask,bid)], data.Ask, data.Bid))
    data['VolumeProfile_normalized'] = data['VolumeProfile'].apply(lambda x: normalize_profile(x,sum(x)))
    data['VolumeProfile_interpolated'] = data['VolumeProfile_normalized'].apply(lambda x: interpolate_profile(x,num_intervals), num_intervals) 
    data = pd.concat([data.drop(['VolumeProfile_interpolated'], axis=1), data['VolumeProfile_interpolated'].apply(pd.Series).add_prefix('VP')], axis=1)
    return data

## Funciones basicas del pretramiento
def candle_volume_profile(df) -> pd.DataFrame:
    df['VolumeProfile']=list(map(lambda ask,bid: [ int(a)+int(b) for a,b in zip(ask,bid)], df.Ask,df.Bid))
    return df

def candle_delta_profile(df) -> pd.DataFrame:
    df['DeltaProfile']=list(map(lambda ask,bid: [ int(a)-int(b) for a,b in zip(ask,bid)], df.Ask,df.Bid))
    return df

# --- Normalizacion  ---
# Funcion que realiza la descomposicion en seno y coseno del tiempo
def sincostime(data: pd.DataFrame) -> pd.DataFrame:
    data['hour'] = pd.to_datetime(data['Time']).dt.hour 
    data['minute'] = pd.to_datetime(data['Time']).dt.minute 
    data['sin_time'] = np.sin(2 * np.pi * (data['hour'] * 60 + data['minute']) / 1440) 
    data['cos_time'] = np.cos(2 * np.pi * (data['hour'] * 60 + data['minute']) / 1440) 
    return data

# Funcion que calcula y pretrata el target
def target_pretreatment(data: pd.DataFrame, target: str, period: int=10, digits: int=10, ticksize: float=0.25) -> pd.DataFrame:
    if target == 'High':
        data['Target'] = data['High'] - data['Open']
    else:
        data['Target'] = data['Open'] - data['Low']
    data['Target_ticks'] = data['Target']/ticksize
    data['Target_nor'] = data['Target_ticks'] / 100
    return data


def features_ma(df, period: int,features:list) -> pd.DataFrame:
    for feature in features:
        df[f'{feature}_MA{period}'] = df[feature].rolling(10).mean()
    return df

def features_pct_change(df,period,features:list):
    for feature in features:
        df[f'{feature}_pct'] = df[f'{feature}_MA{period}'].pct_change()
    return df

def features_normalized(df:pd.DataFrame, features: list, min_value: float, max_value:float,prev: str='pct') -> pd.DataFrame:
    for feature in features:
        df[f'{feature}_nor'] =(df[f'{feature}_{prev}'] - min_value) / (max_value - min_value)
    return df


def index_sets(data: pd.DataFrame,test_pct: float, val_pct: float) -> dict:
    times = sorted(data.index.values)
    start_test = sorted(data.index.values)[-int(test_pct*len(times))] 
    start_val = sorted(data.index.values)[-int(val_pct*len(times))] 
    return {'start_test': start_test, 'start_val': start_val}

# Funcion que devuelve los valores de normalizacion de los datos
def normalization_values(data: pd.DataFrame, index_sets: dict) -> dict:
    # Normaliazamos las columnas de precios para los datos de entrenamiento
    min_price = min(data[(data.index < index_sets['start_val'])][['Open_pct', 'High_pct', 'Low_pct', 'Close_pct']].min(axis=0))
    max_price = max(data[(data.index < index_sets['start_val'])][['Open_pct', 'High_pct', 'Low_pct', 'Close_pct']].max(axis=0))
    
    # Normalizamos las columnas de volumen para los datos de entrenamiento
    if 'Volume_pct' in data.columns:
        min_volume = data[(data.index < index_sets['start_val'])]['Volume_pct'].min(axis=0)
        max_volume = data[(data.index < index_sets['start_val'])]['Volume_pct'].max(axis=0)
    else:
        min_volume = None
        max_volume = None

    # Normalizamos las columnas de delta para los datos de entrenamiento
    if 'Delta_MA10' in data.columns:
        min_delta = data[(data.index < index_sets['start_val'])]['Delta_MA10'].min(axis=0)
        max_delta = data[(data.index < index_sets['start_val'])]['Delta_MA10'].max(axis=0)
    else:
        min_delta = None
        max_delta = None

    return {'min_price': min_price, 'max_price': max_price, 'min_volume': min_volume, 'max_volume': max_volume, 'min_delta': min_delta, 'max_delta': max_delta}
 

def whole_pretreatment(data: pd.DataFrame)->dict:
    # Time
    data = sincostime(data)

    # Moving Average
    period = 10
    data =features_ma(data,period,features=['Open','High','Low','Close','MVC','Volume','Delta'])

    # PCT Change
    data = features_pct_change(data,period,features=['Open','High','Low','Close','MVC','Volume'])

    # Volume Profile
    data = volume_profile(data)

    # Indices de los sets de entrenamiento, validacion y test
    index = index_sets(data, 0.1, 0.2)

    normalized_values = normalization_values(data,index)

    data = features_normalized(data,['Open','High','Low','Close','MVC'],normalized_values['min_price'],normalized_values['max_price'])
    data = features_normalized(data,['Volume'],normalized_values['min_volume'],normalized_values['max_volume'])
    data = features_normalized(data,['Delta'],normalized_values['min_delta'],normalized_values['max_delta'],prev='MA10')


    # Eliminamos las filas con NaN
    data.dropna(how='any', axis=0, inplace=True) 

    return {'data':data, 'normalized_values': normalized_values}


#Funcion que muestras los histogramas de los perfiles de volumenprofilr y deltaprofile de una vela
def show_candle_vp_dp(vp: pd.Series, dp: pd.Series, i: int) -> None:
    plt.subplot(1, 2, 1)
    plt.bar(range(len(vp[i])), vp[i], align='center', alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.bar(range(len(dp[i])), dp[i], align='center', alpha=0.5)

