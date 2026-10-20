# Radar de Tendencias Delictivas - Tablero
# TPI Introduccion a la Ciencia de Datos - Grupo 10 - UGR

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Radar de Tendencias Delictivas', layout='wide')

# 1. Datos
# Nombres cortos de los delitos (en el dataset son muy largos)
DELITOS = {
    'Hurtos': 'Hurtos',
    'Robos (excluye los agravados por el resultado de lesiones y/o muertes)': 'Robos',
    'Robos agravados por el resultado de lesiones y/o muertes': 'Robos agravados',
}

# Nombres cortos de algunas provincias para los graficos
CORTO = {'Ciudad Autónoma de Buenos Aires': 'CABA',
         'Tierra del Fuego, Antártida e Islas del Atlántico Sur': 'Tierra del Fuego'}

# Colores del semaforo
COLORES = {'Rojo': '#d03b3b', 'Amarillo': '#fab219', 'Verde': '#0ca30c'}

@st.cache_data
def cargar_datos():
    # Es el mismo archivo que exportamos en el notebook (capitulo 2)
    df = pd.read_csv('Dataset_snic_limpio.csv')
    df['delito'] = df['codigo_delito_snic_nombre'].replace(DELITOS)
    df['provincia'] = df['provincia_nombre'].replace(CORTO)

    # Poblacion = hechos / tasa * 100.000 (como en el notebook)
    df['poblacion'] = df['cantidad_hechos'] / df['tasa_hechos'] * 100000
    # En las filas con 0 hechos no se puede calcular, asi que usamos la de los otros delitos
    df['poblacion'] = df.groupby(['provincia_nombre', 'anio'])['poblacion'].transform('max')
    return df

@st.cache_data
def cargar_mapa():
    with open('provincias_argentina.geojson', encoding='utf-8') as archivo:
        return json.load(archivo)

df = cargar_datos()
mapa = cargar_mapa()

# 2. Filtros (barra lateral)
st.sidebar.header('Filtros')

delitos_elegidos = st.sidebar.multiselect('Tipos de delito',
                                          ['Hurtos', 'Robos', 'Robos agravados'],
                                          default=['Hurtos', 'Robos', 'Robos agravados'])
if len(delitos_elegidos) == 0:
    st.warning('Elegí al menos un tipo de delito.')
    st.stop()

anio = st.sidebar.selectbox('Año a comparar con la prepandemia', [2025, 2024, 2023, 2022, 2021])

st.sidebar.caption('Nivel prepandemia = promedio 2017-2019. '
                   'El 2020 queda afuera por la cuarentena (ASPO).')
st.sidebar.caption('Fuente: SNIC - Ministerio de Seguridad de la Nación.')


# 3. Calculos (los mismos pasos que el capitulo 5 del notebook)
datos = df[df['delito'].isin(delitos_elegidos)]

# Tasa por provincia y año, sumando los delitos elegidos
tasa = datos.pivot_table(index='provincia', columns='anio', values='tasa_hechos', aggfunc='sum')

tabla = pd.DataFrame()
tabla['pre'] = tasa[[2017, 2018, 2019]].mean(axis=1)   # x: nivel prepandemia
tabla['actual'] = tasa[anio]                             # y: año elegido

# Regresion lineal simple (minimos cuadrados)
b = tabla['pre'].cov(tabla['actual']) / tabla['pre'].var()
a = tabla['actual'].mean() - b * tabla['pre'].mean()
r2 = tabla['pre'].corr(tabla['actual']) ** 2

tabla['esperado'] = a + b * tabla['pre']
tabla['residuo'] = tabla['actual'] - tabla['esperado']
tabla['variacion'] = (tabla['actual'] / tabla['pre'] - 1) * 100


def semaforo(residuo, actual, pre):
    if residuo > 0 and actual > pre:
        return 'Rojo'
    elif residuo > 0:
        return 'Amarillo'
    else:
        return 'Verde'

colores = []
for provincia in tabla.index:
    fila = tabla.loc[provincia]
    colores.append(semaforo(fila['residuo'], fila['actual'], fila['pre']))
tabla['semaforo'] = colores

# Volvemos al nombre oficial para poder ubicar cada provincia en el mapa
nombre_oficial = {}
for nombre_largo, nombre_corto in CORTO.items():
    nombre_oficial[nombre_corto] = nombre_largo
tabla = tabla.reset_index()
tabla['provincia_nombre'] = tabla['provincia'].replace(nombre_oficial)

# Tasa nacional = hechos del pais / poblacion del pais * 100.000
hechos_pais = datos.groupby('anio')['cantidad_hechos'].sum()
poblacion_pais = df.drop_duplicates(['provincia_nombre', 'anio']).groupby('anio')['poblacion'].sum()
nacional = hechos_pais / poblacion_pais * 100000


def numero(valor, decimales=0):
    # Formato argentino: punto para miles y coma para decimales
    texto = f'{valor:,.{decimales}f}'
    return texto.replace(',', 'X').replace('.', ',').replace('X', '.')

# 4. Encabezado e indicadores
st.title('Radar de Tendencias Delictivas')
st.write(f'Delitos contra la propiedad ({", ".join(delitos_elegidos)}). '
         f'Comparación {anio} vs. nivel prepandemia (2017-2019). Tasas cada 100.000 habitantes.')

pre_nacional = nacional[[2017, 2018, 2019]].mean()
cambio_anual = (nacional[anio] / nacional[anio - 1] - 1) * 100
cambio_pre = (nacional[anio] / pre_nacional - 1) * 100
cantidad_rojas = (tabla['semaforo'] == 'Rojo').sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric(f'Tasa nacional {anio}', numero(nacional[anio]),
            f'{numero(cambio_anual, 1)}% vs. {anio - 1}', delta_color='inverse')
col2.metric('Tasa nacional prepandemia', numero(pre_nacional),
            f'{numero(cambio_pre, 1)}% en {anio}', delta_color='inverse')
col3.metric('Provincias en rojo', f'{cantidad_rojas} de 24')
col4.metric('R² del modelo', numero(r2, 2))

if r2 < 0.3:
    st.warning('Con esta selección el nivel prepandemia explica muy poco (R² bajo), '
               'así que el semáforo no es confiable. Pasa por ejemplo con robos agravados solos.')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Mapa', 'Modelo', 'Ranking', 'Evolución', 'Metodología'])

# 5. Pestaña Mapa
with tab1:
    opcion = st.radio('Colorear por', ['Semáforo', 'Variación vs. prepandemia', f'Tasa {anio}'],
                      horizontal=True)

    datos_hover = {'pre': ':.0f', 'actual': ':.0f', 'esperado': ':.0f',
                   'residuo': ':.0f', 'variacion': ':.1f', 'provincia_nombre': False}
    etiquetas = {'pre': 'Prepandemia', 'actual': f'Tasa {anio}', 'esperado': 'Esperado',
                 'residuo': 'Residuo', 'variacion': 'Variación vs pre (%)', 'semaforo': 'Semáforo'}

    if opcion == 'Semáforo':
        color = 'semaforo'
    elif opcion == 'Variación vs. prepandemia':
        color = 'variacion'
    else:
        color = 'actual'

    fig = px.choropleth_map(tabla, geojson=mapa, locations='provincia_nombre',
                            featureidkey='properties.provincia_nombre', color=color,
                            color_discrete_map=COLORES,
                            category_orders={'semaforo': ['Rojo', 'Amarillo', 'Verde']},
                            color_continuous_scale='Blues',
                            hover_name='provincia', hover_data=datos_hover, labels=etiquetas,
                            map_style='white-bg', center={'lat': -40.5, 'lon': -63.5},
                            zoom=3.1, opacity=0.9)
    if opcion == 'Variación vs. prepandemia':
        limite = tabla['variacion'].abs().max()
        fig.update_layout(coloraxis=dict(colorscale='RdBu_r', cmin=-limite, cmax=limite))
    fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0))

    izquierda, derecha = st.columns([3, 2])
    with izquierda:
        st.plotly_chart(fig, width='stretch', config={'scrollZoom': True})
    with derecha:
        st.subheader('Cómo leer el semáforo')
        st.write('**Rojo:** peor de lo esperado y por encima de su nivel prepandemia.')
        st.write('**Amarillo:** peor de lo esperado, pero por debajo de su nivel previo.')
        st.write('**Verde:** igual o mejor de lo esperado.')
        st.write('El valor **esperado** es el que da la recta de regresión según el nivel '
                 'prepandemia de cada provincia. El **residuo** es la diferencia entre el valor '
                 'real y el esperado.')
        for color_sem in ['Rojo', 'Amarillo']:
            lista = tabla[tabla['semaforo'] == color_sem].sort_values('residuo', ascending=False)
            st.write(f'**{color_sem}:** ' + ', '.join(lista['provincia']))
        st.caption('Se puede hacer zoom con la rueda del mouse. CABA es muy chica en el mapa, '
                   'conviene verla en el Ranking.')

# 6. Pestaña Modelo
with tab2:
    st.write(f'**Recta de regresión:** tasa {anio} = {numero(a)} + {numero(b, 2)} × tasa prepandemia'
             f'  |  **R² = {numero(r2, 2)}**  |  24 provincias')

    maximo = max(tabla['pre'].max(), tabla['actual'].max()) * 1.08

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, maximo], y=[0, maximo], mode='lines',
                             line=dict(color='gray', dash='dash'),
                             name='y = x (igual que antes de la pandemia)'))
    fig.add_trace(go.Scatter(x=[0, maximo], y=[a, a + b * maximo], mode='lines',
                             line=dict(color='#2a78d6', width=3), name='Recta de regresión'))

    for color_sem in ['Rojo', 'Amarillo', 'Verde']:
        parte = tabla[tabla['semaforo'] == color_sem]
        # Solo escribimos el nombre de las provincias mas importantes para que no se amontonen
        nombres = []
        for i in parte.index:
            if parte.loc[i, 'semaforo'] != 'Verde' or abs(parte.loc[i, 'residuo']) > 150:
                nombres.append(parte.loc[i, 'provincia'])
            else:
                nombres.append('')
        fig.add_trace(go.Scatter(x=parte['pre'], y=parte['actual'], mode='markers+text',
                                 text=nombres, textposition='top center', name=color_sem,
                                 hovertext=parte['provincia'],
                                 marker=dict(size=12, color=COLORES[color_sem])))

    fig.update_layout(height=620, xaxis_title='Tasa promedio 2017-2019 (x)',
                      yaxis_title=f'Tasa {anio} (y)', xaxis_range=[0, maximo],
                      yaxis_range=[0, maximo], legend=dict(orientation='h', y=-0.15))
    st.plotly_chart(fig)

    ordenada = tabla.sort_values('residuo')
    fig2 = px.bar(ordenada, x='residuo', y='provincia', orientation='h', color='semaforo',
                  color_discrete_map=COLORES,
                  category_orders={'provincia': list(ordenada['provincia'])[::-1],
                                   'semaforo': ['Rojo', 'Amarillo', 'Verde']},
                  labels={'residuo': f'Residuo (tasa real {anio} - tasa esperada)', 'provincia': ''},
                  title='Distancia a lo esperado (positivo = peor de lo esperado)')
    fig2.update_layout(height=650)
    st.plotly_chart(fig2)

# 7. Pestaña Ranking
with tab3:
    ranking = tabla.sort_values('residuo', ascending=False)
    ranking = ranking[['provincia', 'pre', 'actual', 'esperado', 'residuo', 'variacion', 'semaforo']]
    ranking.columns = ['Provincia', 'Prepandemia (2017-19)', f'Tasa {anio}', 'Esperado',
                       'Residuo', 'Variación vs pre (%)', 'Semáforo']
    ranking = ranking.round(1)
    st.dataframe(ranking, hide_index=True, height=880)
    st.download_button('Descargar tabla (CSV)', ranking.to_csv(index=False), f'radar_{anio}.csv')

# 8. Pestaña Evolucion
with tab4:
    c1, c2 = st.columns([2, 1])
    mas_altas = list(tabla.sort_values('residuo', ascending=False)['provincia'][:3])
    elegidas = c1.multiselect('Provincias', sorted(tabla['provincia']), default=mas_altas)
    desde, hasta = c2.slider('Período', 2000, 2025, (2010, 2025))

    serie = datos[datos['provincia'].isin(elegidas)]
    serie = serie[(serie['anio'] >= desde) & (serie['anio'] <= hasta)]
    serie = serie.groupby(['provincia', 'anio'], as_index=False)['tasa_hechos'].sum()

    fig = px.line(serie, x='anio', y='tasa_hechos', color='provincia', markers=True,
                  color_discrete_sequence=px.colors.qualitative.Set1,
                  labels={'anio': 'Año', 'tasa_hechos': 'Tasa cada 100.000 hab.', 'provincia': 'Provincia'})
    pais = nacional.loc[desde:hasta]
    fig.add_trace(go.Scatter(x=pais.index, y=pais.values, mode='lines', name='Argentina (total)',
                             line=dict(color='black', dash='dot')))
    if desde <= 2020 <= hasta:
        fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor='gray', opacity=0.15, line_width=0,
                      annotation_text='Pandemia')
    fig.update_layout(height=520)
    st.plotly_chart(fig)
    st.caption('Por defecto se ven las 3 provincias con mayor residuo. '
               'La línea punteada es la tasa nacional.')

# 9. Pestaña Metodologia
with tab5:
    st.subheader('Qué responde el tablero')
    st.write(f'¿En qué provincias la tasa de delitos contra la propiedad de {anio} quedó por '
             'encima de lo que su propio nivel prepandemia permitía esperar, y cuáles ya superan '
             'ese nivel previo?')

    st.subheader('Cómo se calcula')
    st.write('1. Datos del SNIC por provincia, filtrados a hurtos, robos y robos agravados '
             '(1.872 filas, 2000-2025).')
    st.write('2. Tasa = hechos cada 100.000 habitantes. Se suman las tasas de los delitos elegidos.')
    st.write(f'3. x = promedio 2017-2019 (prepandemia). y = tasa {anio}.')
    st.write('4. Regresión lineal simple entre las 24 provincias (mínimos cuadrados).')
    st.write('5. Residuo = valor real - valor esperado. Positivo = peor de lo esperado.')
    st.write('6. Semáforo: rojo si el residuo es positivo y además supera su nivel prepandemia; '
             'amarillo si solo el residuo es positivo; verde en el resto.')

    st.subheader('Limitaciones')
    st.write('- Son solo 24 provincias: un caso extremo puede mover la recta (por ejemplo CABA o Neuquén).')
    st.write('- Es delito registrado: depende de las denuncias y de cómo registra cada provincia.')
    st.write('- Muestra una relación, no explica causas ni predice el futuro.')
