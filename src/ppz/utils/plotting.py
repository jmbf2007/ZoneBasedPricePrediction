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

from MLfunctions_predictions import normal_distribution
  



# -------- Grafiicos de Matplotlib --------
def plot_candlestick_predictions(df1:pd.DataFrame, df2:pd.DataFrame=None,name1: str='5M', name2: str='15M',width: int=2000, heigth: int=1600) -> object:
    """
    Funcion que dibuja en el mismo grafico dos series de velas japonesas de distinta temporalidad y sus predicciones
    """
    fig = go.Figure()

    # Agregar las velas de la serie de 5 minutos
    fig.add_trace(go.Candlestick(x=df1['Time'],
                    open=df1['Open'],
                    high=df1['High'],
                    low=df1['Low'],
                    close=df1['Close'],
                    name=name1,
                    increasing_line={'color':'green'},  # Establecer el grosor de la línea para las velas al alza
                    decreasing_line={'color':'red'} # Establecer el grosor de la línea para las velas a la baja
                    ))


    # Añadir predicciones de las series de 5 y 15 minutos
    if 'Pred_High' in df1.columns:
        fig.add_trace(go.Scatter(x=df1['Time'], y=df1['Pred_High'], mode='lines+markers', name='Pred_High', line=dict(color='royalblue', width=2)))
    if 'Pred_Low' in df1.columns:
        fig.add_trace(go.Scatter(x=df1['Time'], y=df1['Pred_Low'], mode='lines+markers', name='Pred_Low', line=dict(color='firebrick', width=2)))
    if 'Pred_Close' in df1.columns:
        fig.add_trace(go.Scatter(x=df1['Time'], y=df1['Pred_Close'], mode='lines+markers', name='Pred_Close', line=dict(color='orange', width=2)))

    title = f'Predicciones de la serie de {name1}'
    
    # Agregar las velas de la serie de 15 minutos con mayor transparencia
    if df2 is not None:
        fig.add_trace(go.Candlestick(x=df2['Time'],
                        open=df2['Open'],
                        high=df2['High'],
                        low=df2['Low'],
                        close=df2['Close'],
                        name=name2,
                        increasing={'line': {'color': 'rgba(0, 0, 255, 0.5)'}}, # Ajustar opacidad para velas al alza
                        decreasing={'line': {'color': 'rgba(255, 0, 0, 0.5)'}})) # Ajustar opacidad para velas a la baja
        
        if 'Pred_High' in df2.columns:
            fig.add_trace(go.Scatter(x=df2['Time'], y=df2['Pred_High'], mode='lines+markers', name='Pred_High', line=dict(color='royalblue', width=2, dash='dash')))
        if 'Pred_Low' in df2.columns:
            fig.add_trace(go.Scatter(x=df2['Time'], y=df2['Pred_Low'], mode='lines+markers', name='Pred_Low', line=dict(color='pink', width=2, dash='dash')))
        if 'Pred_Close' in df2.columns:
            fig.add_trace(go.Scatter(x=df2['Time'], y=df2['Pred_Close'], mode='lines+markers', name='Pred_Close', line=dict(color='orange', width=2, dash='dash')))
            
        title = f'Predicciones de las series de {name1} y {name2}'

    # Actualizar el layout del gráfico si es necesario
    fig.update_layout(
        width=width,
        height=heigth,
        xaxis_rangeslider_visible=False, 
        title=title
    )
    return fig

def plot_normal_distribution(dif_ticks):
    # Calculamos la media y la desviación estándar de dif_ticks
    media = np.mean(dif_ticks)
    desviacion_estandar = np.std(dif_ticks)

    # Creamos un histograma de dif_ticks
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(dif_ticks, bins=30, density=True, alpha=0.7, color='skyblue')

    # Generamos la curva de distribución normal
    x = np.linspace(min(dif_ticks), max(dif_ticks), 100)
    y = stats.norm.pdf(x, media, desviacion_estandar)
    plt.plot(x, y, 'r-', linewidth=2)

    # Añadimos líneas verticales para la media y las desviaciones estándar
    plt.axvline(media, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(media + desviacion_estandar, color='green', linestyle='dashed', linewidth=2)
    plt.axvline(media - desviacion_estandar, color='green', linestyle='dashed', linewidth=2)

    # Configuramos el gráfico
    plt.title('Distribución de las diferencias en ticks')
    plt.xlabel('Diferencia en ticks')
    plt.ylabel('Densidad')

    # Añadimos una leyenda en el centro a la izquierda
    plt.legend(['Distribución Normal', 'Media', '±1 Desviación Estándar', 'Datos'], loc='center left')


    # Calculamos el porcentaje de datos dentro de 1, 2 y 3 desviaciones estándar
    dentro_1_std = np.sum((dif_ticks >= media - desviacion_estandar) & (dif_ticks <= media + desviacion_estandar)) / len(dif_ticks) * 100
    dentro_2_std = np.sum((dif_ticks >= media - 2*desviacion_estandar) & (dif_ticks <= media + 2*desviacion_estandar)) / len(dif_ticks) * 100
    dentro_3_std = np.sum((dif_ticks >= media - 3*desviacion_estandar) & (dif_ticks <= media + 3*desviacion_estandar)) / len(dif_ticks) * 100

    # Mostramos estadísticas
    line1 = f'Media: {media:.2f}'
    line2 = f'Desviación Estándar: {desviacion_estandar:.2f}'
    line3 = f'% dentro de 1 desviación estándar: {dentro_1_std:.2f}%'
    line4 = f'% dentro de 2 desviaciones estándar: {dentro_2_std:.2f}%'
    line5 = f'% dentro de 3 desviaciones estándar: {dentro_3_std:.2f}%'
    plt.text(0.05, 0.95, line1 + '\n' + line2 + '\n' + line3 + '\n' + line4 + '\n' + line5, 
            transform=plt.gca().transAxes, verticalalignment='top')

    distribution={
        'media': media,
        'desviacion_estandar': desviacion_estandar,
        'dentro_1_std': dentro_1_std,
        'dentro_2_std': dentro_2_std,
        'dentro_3_std': dentro_3_std
    }
    return distribution, plt

def plot_dif_ticks_deviation_points(dif_ticks):
    # Calculamos la media y la desviación estándar de dif_ticks
    media = np.mean(dif_ticks)
    desviacion_estandar = np.std(dif_ticks)

    # Definimos el umbral para considerar una desviación como "mayor"
    umbral = 2 * desviacion_estandar

    # Identificamos las desviaciones mayores
    desviaciones_mayores = [(i, dif) for i, dif in enumerate(dif_ticks) if abs(dif - media) > umbral]
    
    # Visualizamos las desviaciones mayores en la gráfica
    plt.figure(figsize=(12, 6))
    plt.plot(dif_ticks, label='Diferencia en ticks')
    plt.scatter([i for i, _ in desviaciones_mayores], [dif for _, dif in desviaciones_mayores], color='red', label='Desviaciones mayores')
    plt.axhline(media, color='green', linestyle='--', label='Media')
    plt.axhline(media + umbral, color='orange', linestyle='--', label='Umbral superior')
    plt.axhline(media - umbral, color='orange', linestyle='--', label='Umbral inferior')
    plt.title('Diferencia entre el precio predicho y el precio real en ticks')
    plt.xlabel('Índice')
    plt.ylabel('Diferencia en ticks')
    plt.legend()
    return plt

def plot_predictions(real, predictions):
    plt.plot(real)
    plt.plot(predictions)
    plt.legend(['Real','Pred'])
    plt.title('Predicion vs Real')
    return plt

def plot_dispersion_scatter(serie1, serie2, title:str, xlabel:str, ylabel:str):
    plt.figure(figsize=(12, 6))
    plt.scatter(serie1, serie2, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    return plt

### --------------------- Graficos Plotly --------------------- ###

def plotly_normal_distribution(dif_ticks):

    distribution = normal_distribution(dif_ticks)

    # Calculamos la media y la desviación estándar de dif_ticks
    media = distribution['media']
    desviacion_estandar = distribution['desviacion_estandar']

    # Generamos el histograma
    hist_data = go.Histogram(x=dif_ticks, nbinsx=30, histnorm='probability density', opacity=0.7, name='Datos')

    # Generamos la curva de distribución normal
    x = np.linspace(min(dif_ticks), max(dif_ticks), 100)
    y = stats.norm.pdf(x, media, desviacion_estandar)
    curve_data = go.Scatter(x=x, y=y, mode='lines', name='Distribución Normal', line=dict(color='red'))

    # Creamos las líneas de la media y las desviaciones estándar
    lines = [
        go.Scatter(x=[media, media], y=[0, max(y)], mode="lines", line=dict(color="red", dash="dash"), name="Media"),
        go.Scatter(x=[media + desviacion_estandar, media + desviacion_estandar], y=[0, max(y)], mode="lines", line=dict(color="green", dash="dash"), name="+1 Desviación Estándar"),
        go.Scatter(x=[media - desviacion_estandar, media - desviacion_estandar], y=[0, max(y)], mode="lines", line=dict(color="green", dash="dash"), name="-1 Desviación Estándar")
    ]

    # Creamos la figura
    fig = go.Figure(data=[hist_data, curve_data] + lines)

    # Ajustes del layout
    fig.update_layout(
        title="Distribución de las diferencias en ticks",
        xaxis_title="Diferencia en ticks",
        yaxis_title="Densidad",
        showlegend=True
    )
    # Recuadro de texto con los datos de la distribucion en la parte superior izquierda, justificado a la izquierda
    line1 = f'Media: {media:.2f}'
    line2 = f'Desviación Estándar: {desviacion_estandar:.2f}'
    line3 = f'% dentro de 1 desviación estándar: {distribution["dentro_1_std"]:.2f}%'
    line4 = f'% dentro de 2 desviaciones estándar: {distribution["dentro_2_std"]:.2f}%'
    line5 = f'% dentro de 3 desviaciones estándar: {distribution["dentro_3_std"]:.2f}%'

    fig.add_annotation(
        x=0.05,  # Alineación horizontal centrada
        y=0.95,  # Alineación vertical centrada
        xref='paper',  # Referencia de coordenadas del papel
        yref='paper',  # Referencia de coordenadas del papel
        text=line1 + '<br>' + line2 + '<br>' + line3 + '<br>' + line4 + '<br>' + line5,
        showarrow=False,
        font=dict(size=12),
        align='left'
    )
    return fig

def plotly_predictions(real, predictions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(len(real))), y=real, mode='lines', name='Real'))
    fig.add_trace(go.Scatter(x=list(range(len(predictions))), y=predictions, mode='lines', name='Pred'))

    fig.update_layout(
        title="Predicción vs Real",
        xaxis_title="Índice",
        yaxis_title="Valor",
        showlegend=True
    )

    return fig

def plotly_dif_ticks_deviation_points_with_ATR(dif_ticks, atr_series):
    # Calculamos la media y la desviación estándar de dif_ticks
    media = np.mean(dif_ticks)
    desviacion_estandar = np.std(dif_ticks)

    # Definimos el umbral
    umbral = 2 * desviacion_estandar

    # Identificamos las desviaciones mayores
    desviaciones_mayores = [(i, dif) for i, dif in enumerate(dif_ticks) if abs(dif - media) > umbral]

    # Creamos la figura con dos subplots que comparten el mismo eje X
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Diferencia en ticks", "ATR (Average True Range)"))

    # Gráfico 1: Línea de las diferencias en ticks
    fig.add_trace(go.Scatter(x=list(range(len(dif_ticks))), y=dif_ticks, mode='lines', name='Diferencia en ticks'), row=1, col=1)

    # Puntos de desviaciones mayores
    fig.add_trace(go.Scatter(x=[i for i, _ in desviaciones_mayores], 
                             y=[dif for _, dif in desviaciones_mayores], 
                             mode='markers', marker=dict(color='red'), name='Desviaciones mayores'), row=1, col=1)

    # Líneas de media y umbrales
    fig.add_trace(go.Scatter(x=[0, len(dif_ticks)], y=[media, media], mode='lines', name='Media', line=dict(dash='dash', color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, len(dif_ticks)], y=[media + umbral, media + umbral], mode='lines', name='Umbral superior', line=dict(dash='dash', color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, len(dif_ticks)], y=[media - umbral, media - umbral], mode='lines', name='Umbral inferior', line=dict(dash='dash', color='orange')), row=1, col=1)

    # Gráfico 2: ATR
    fig.add_trace(go.Scatter(x=list(range(len(atr_series))), y=atr_series, mode='lines', name='ATR', line=dict(color='firebrick')), row=2, col=1)

    # Ajustamos el layout para ambos gráficos
    fig.update_layout(
        title="Diferencia entre el precio predicho y el precio real en ticks y ATR",
        xaxis_title="Índice",
        yaxis_title="Diferencia en ticks",
        showlegend=True,
        height=800
    )

    # Ajustar el título del segundo gráfico (ATR)
    fig.update_yaxes(title_text="ATR", row=2, col=1)

    return fig

def plotly_dispersion_scatter(serie1, serie2, title:str, xlabel:str, ylabel:str):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=serie1, y=serie2, mode='markers', opacity=0.5))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=False
    )

    return fig

def plotly_mosaic(fig1, fig2, fig3, fig4):
    # Crear un layout 2x2
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Real vs Prediction", "Diferencia Ticks (Real - Prediccion)", "ATR vs Diferencia de Ticks", "Distribucion de las Diferencias Ticks"))

    # Añadir gráficos al mosaico y ajustar las leyendas
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)
    
    fig.update_xaxes(title_text="Índice", row=1, col=1)
    fig.update_yaxes(title_text="Valor", row=1, col=1)
    
    for trace in fig2['data']:
        fig.add_trace(trace, row=1, col=2)
    
    fig.update_xaxes(title_text="Índice", row=1, col=2)
    fig.update_yaxes(title_text="Diferencia en ticks", row=1, col=2)
    
    for trace in fig3['data']:
        fig.add_trace(trace, row=2, col=1)
    
    fig.update_xaxes(title_text="ATR", row=2, col=1)
    fig.update_yaxes(title_text="Diferencia de ticks", row=2, col=1)
    
    for trace in fig4['data']:
        fig.add_trace(trace, row=2, col=2)
    
    fig.update_xaxes(title_text="Diferencia en ticks", row=2, col=2)
    fig.update_yaxes(title_text="Densidad", row=2, col=2)

    # Ajustar la posición de la leyenda en cada gráfico
    fig.update_layout(
        showlegend=True,
        height=800, 
        width=1000,
        title_text="Mosaico de Gráficos"
    )

    # Configurar la leyenda de cada gráfico al lado derecho
    fig.update_layout(legend=dict(x=1.05, y=0.5))

    return fig