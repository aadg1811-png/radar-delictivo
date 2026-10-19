"""
Radar de Tendencias Delictivas · Tablero interactivo
TPI Introducción a la Ciencia de Datos · Grupo 10 · UGR

Cómo ejecutarlo (desde la carpeta donde está este archivo):
    pip install -r requirements.txt
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CARPETA = Path(__file__).parent
ARCHIVO_DATOS = CARPETA / 'Dataset_snic_limpio.csv'      # exportado por el notebook (Cap. 2)
ARCHIVO_MAPA = CARPETA / 'provincias_argentina.geojson'

DELITOS = {
    'Hurtos': 'Hurtos',
    'Robos (excluye los agravados por el resultado de lesiones y/o muertes)': 'Robos',
    'Robos agravados por el resultado de lesiones y/o muertes': 'Robos agravados',
}
CORTO = {'Ciudad Autónoma de Buenos Aires': 'CABA',
         'Tierra del Fuego, Antártida e Islas del Atlántico Sur': 'Tierra del Fuego'}
PRE = [2017, 2018, 2019]

SEMAFORO = {
    'Rojo':     {'color': '#d03b3b', 'texto': '🔴 Rojo · peor de lo esperado y sobre su nivel previo'},
    'Amarillo': {'color': '#fab219', 'texto': '🟡 Amarillo · peor de lo esperado'},
    'Verde':    {'color': '#0ca30c', 'texto': '🟢 Verde · igual o mejor de lo esperado'},
}
AZUL, GRIS = '#2a78d6', '#898781'
PALETA = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

st.set_page_config(page_title='Radar de Tendencias Delictivas', page_icon='🗺️', layout='wide')


# ───────────────────────────── Datos ─────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv(ARCHIVO_DATOS)
    df['delito'] = df['codigo_delito_snic_nombre'].map(DELITOS)
    df['poblacion'] = df['cantidad_hechos'] / df['tasa_hechos'].replace(0, np.nan) * 100_000
    # La población es la misma para los 3 delitos de una provincia-año: la completamos para los ceros
    df['poblacion'] = df.groupby(['provincia_nombre', 'anio'])['poblacion'].transform('max')
    df['provincia'] = df['provincia_nombre'].replace(CORTO)
    return df


@st.cache_data
def cargar_mapa():
    return json.loads(ARCHIVO_MAPA.read_text(encoding='utf-8'))


def armar_radar(df, delitos, anio):
    """Tabla de 24 provincias con x (prepandemia), y (año elegido), regresión y semáforo."""
    sel = df[df['delito'].isin(delitos)]
    tasa = sel.pivot_table(index=['provincia_nombre', 'provincia'], columns='anio',
                           values='tasa_hechos', aggfunc='sum').reset_index('provincia')
    t = pd.DataFrame({'provincia': tasa['provincia'],
                      'pre': tasa[PRE].mean(axis=1),
                      'actual': tasa[anio]})
    b, a = np.polyfit(t['pre'], t['actual'], 1)            # mínimos cuadrados: actual = a + b·pre
    r2 = np.corrcoef(t['pre'], t['actual'])[0, 1] ** 2
    t['esperado'] = a + b * t['pre']
    t['residuo'] = t['actual'] - t['esperado']
    t['var_pre'] = ((t['actual'] / t['pre'] - 1) * 100).replace([np.inf, -np.inf], np.nan)
    t['semaforo'] = np.select([(t['residuo'] > 0) & (t['actual'] > t['pre']), t['residuo'] > 0],
                              ['Rojo', 'Amarillo'], 'Verde')
    return t.reset_index(), a, b, r2


def tasa_nacional(df, delitos):
    sel = df[df['delito'].isin(delitos)]
    hechos = sel.groupby('anio')['cantidad_hechos'].sum()
    pob = df.drop_duplicates(['provincia_nombre', 'anio']).groupby('anio')['poblacion'].sum()
    return hechos / pob * 100_000


def fmt(v, dec=0):
    return f'{v:,.{dec}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


if not ARCHIVO_DATOS.exists():
    st.error('No encuentro **Dataset_snic_limpio.csv**. Copialo en la misma carpeta que app.py '
             '(lo genera el notebook, en el Capítulo 2).')
    st.stop()

df = cargar_datos()
geo = cargar_mapa() if ARCHIVO_MAPA.exists() else None

# ───────────────────────────── Filtros ─────────────────────────────
with st.sidebar:
    st.header('Filtros')
    delitos = st.multiselect('Tipos de delito', list(DELITOS.values()), default=list(DELITOS.values()))
    if not delitos:
        st.warning('Elegí al menos un tipo de delito.')
        st.stop()
    anio = st.selectbox('Año a comparar con la prepandemia', [2025, 2024, 2023, 2022, 2021], index=0)
    st.caption('Nivel prepandemia = promedio 2017-2019. El año 2020 queda fuera por el efecto del ASPO.')
    st.divider()
    st.caption('Fuente: SNIC – Ministerio de Seguridad de la Nación (snic-provincias.xlsx).\n\n'
               'TPI Introducción a la Ciencia de Datos · Grupo 10 · UGR')

radar, a, b, r2 = armar_radar(df, delitos, anio)
nac = tasa_nacional(df, delitos)

# ───────────────────────────── Encabezado ─────────────────────────────
st.title('🗺️ Radar de Tendencias Delictivas')
st.markdown(f'Delitos contra la propiedad (**{", ".join(delitos)}**) · comparación **{anio}** vs. nivel '
            'prepandemia (2017-2019) · tasas cada 100.000 habitantes')

k1, k2, k3, k4 = st.columns(4)
k1.metric(f'Tasa nacional {anio}', fmt(nac[anio]),
          f'{fmt((nac[anio] / nac[anio - 1] - 1) * 100, 1)}% vs. {anio - 1}', delta_color='inverse')
k2.metric('Tasa nacional prepandemia', fmt(nac[PRE].mean()),
          f'{fmt((nac[anio] / nac[PRE].mean() - 1) * 100, 1)}% en {anio}', delta_color='inverse')
k3.metric('Provincias en rojo', f'{(radar["semaforo"] == "Rojo").sum()} de 24')
k4.metric('R² del modelo', fmt(r2, 2), help='Proporción de las diferencias entre provincias '
          f'en {anio} que explica su nivel prepandemia.')

if r2 < 0.3:
    st.warning(f'Con esta selección el nivel prepandemia explica muy poco (R² = {fmt(r2, 2)}): la recta no '
               'describe bien a las provincias y el semáforo no es confiable. Pasa, por ejemplo, con robos '
               'agravados solos, que son pocos casos y con cambios de registro.')

tab_mapa, tab_modelo, tab_ranking, tab_evol, tab_metodo = st.tabs(
    ['🗺️ Mapa', '📈 Modelo', '📋 Ranking', '⏱️ Evolución', 'ℹ️ Metodología'])

# ───────────────────────────── Mapa ─────────────────────────────
with tab_mapa:
    if geo is None:
        st.warning('Falta el archivo provincias_argentina.geojson en la carpeta de la app.')
    else:
        modo = st.radio('Colorear por', ['Semáforo', 'Variación vs. prepandemia', f'Tasa {anio}'],
                        horizontal=True)
        hover = {'provincia': True, 'pre': ':.0f', 'actual': ':.0f', 'esperado': ':.0f',
                 'residuo': ':+.0f', 'var_pre': ':+.1f', 'provincia_nombre': False}
        etiquetas = {'pre': 'Prepandemia', 'actual': f'Tasa {anio}', 'esperado': 'Esperado',
                     'residuo': 'Residuo', 'var_pre': 'Var. vs pre (%)', 'semaforo': 'Semáforo'}
        comunes = dict(geojson=geo, locations='provincia_nombre', featureidkey='properties.provincia_nombre',
                       hover_name='provincia', hover_data=hover, labels=etiquetas,
                       map_style='white-bg', center={'lat': -40.5, 'lon': -63.5}, zoom=3.1, opacity=0.9)
        if modo == 'Semáforo':
            fig = px.choropleth_map(radar, color='semaforo',
                                color_discrete_map={k: v['color'] for k, v in SEMAFORO.items()},
                                category_orders={'semaforo': list(SEMAFORO)}, **comunes)
        elif modo == 'Variación vs. prepandemia':
            lim = radar['var_pre'].abs().max()
            fig = px.choropleth_map(radar, color='var_pre', color_continuous_scale='RdBu_r',
                                range_color=(-lim, lim), **comunes)
        else:
            fig = px.choropleth_map(radar, color='actual', color_continuous_scale='Blues', **comunes)
        fig.update_traces(marker_line_color='white', marker_line_width=0.8)
        fig.update_layout(height=720, margin=dict(l=0, r=0, t=10, b=0), legend_title_text='')

        col_mapa, col_texto = st.columns([3, 2])
        with col_mapa:
            st.plotly_chart(fig, width='stretch', config={'scrollZoom': True})
        with col_texto:
            st.markdown('#### Cómo leer el semáforo')
            for v in SEMAFORO.values():
                st.markdown(f"- {v['texto']}")
            st.markdown('**"Esperado"** es lo que predice la recta de regresión según el nivel prepandemia '
                        'de cada provincia. El **residuo** es la diferencia entre lo real y lo esperado.')
            for sem in ['Rojo', 'Amarillo']:
                prov = radar[radar['semaforo'] == sem].sort_values('residuo', ascending=False)['provincia']
                st.markdown(f"**{SEMAFORO[sem]['texto'].split(' · ')[0]}:** {', '.join(prov) or '—'}")
            st.caption('Podés hacer zoom con la rueda del mouse. CABA es muy chica: acercate a la zona del AMBA o mirala en el Ranking.')

# ───────────────────────────── Modelo ─────────────────────────────
with tab_modelo:
    st.markdown(f'**Recta de regresión:** tasa {anio} = {fmt(a)} + {fmt(b, 2)} × tasa prepandemia '
                f'  ·   **R² = {fmt(r2, 2)}**   ·   n = 24 provincias')
    lim = np.array([0, max(radar['pre'].max(), radar['actual'].max()) * 1.08])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lim, y=lim, mode='lines', line=dict(color=GRIS, dash='dash', width=1),
                             name='y = x (igual que antes de la pandemia)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=lim, y=a + b * lim, mode='lines', line=dict(color=AZUL, width=2.5),
                             name='Recta de regresión', hoverinfo='skip'))
    for sem, v in SEMAFORO.items():
        d = radar[radar['semaforo'] == sem]
        # Nombre visible solo en rojas/amarillas y verdes muy alejadas; el resto aparece al pasar el mouse
        visibles = np.where((d['semaforo'] != 'Verde') | (d['residuo'].abs() > 150), d['provincia'], '')
        fig.add_trace(go.Scatter(
            x=d['pre'], y=d['actual'], mode='markers+text', text=visibles, textposition='top center',
            hovertext=d['provincia'],
            textfont=dict(size=10), name=v['texto'],
            marker=dict(size=12, color=v['color'], line=dict(color='white', width=1.5)),
            customdata=np.stack([d['esperado'], d['residuo']], axis=-1),
            hovertemplate='<b>%{hovertext}</b><br>Prepandemia: %{x:.0f}<br>Real: %{y:.0f}'
                          '<br>Esperado: %{customdata[0]:.0f}<br>Residuo: %{customdata[1]:+.0f}<extra></extra>'))
    fig.update_layout(height=620, xaxis_title='Tasa promedio 2017-2019 (x)', yaxis_title=f'Tasa {anio} (y)',
                      xaxis_range=lim, yaxis_range=lim, legend=dict(orientation='h', y=-0.15),
                      margin=dict(t=10))
    st.plotly_chart(fig, width='stretch')

    res = radar.sort_values('residuo')
    fig2 = px.bar(res, x='residuo', y='provincia', orientation='h', color='semaforo',
                  color_discrete_map={k: v['color'] for k, v in SEMAFORO.items()},
                  category_orders={'provincia': res['provincia'].tolist()[::-1], 'semaforo': list(SEMAFORO)},
                  labels={'residuo': f'Residuo = tasa real {anio} − tasa esperada', 'provincia': '',
                          'semaforo': 'Semáforo'})
    fig2.update_layout(height=650, margin=dict(t=30), title='Distancia a lo esperado (positivo = peor de lo esperado)')
    st.plotly_chart(fig2, width='stretch')

# ───────────────────────────── Ranking ─────────────────────────────
with tab_ranking:
    tabla = (radar.sort_values('residuo', ascending=False)
                  .assign(Semáforo=lambda d: d['semaforo'].map({k: v['texto'].split(' · ')[0] for k, v in SEMAFORO.items()}))
                  [['provincia', 'pre', 'actual', 'esperado', 'residuo', 'var_pre', 'Semáforo']]
                  .rename(columns={'provincia': 'Provincia', 'pre': 'Prepandemia (2017-19)',
                                   'actual': f'Tasa {anio}', 'esperado': 'Esperado',
                                   'residuo': 'Residuo', 'var_pre': 'Var. vs pre (%)'}))
    st.dataframe(tabla, hide_index=True, width='stretch', height=880,
                 column_config={c: st.column_config.NumberColumn(format='%.0f')
                                for c in ['Prepandemia (2017-19)', f'Tasa {anio}', 'Esperado', 'Residuo']}
                 | {'Var. vs pre (%)': st.column_config.NumberColumn(format='%+.1f')})
    st.download_button('Descargar tabla (CSV)', tabla.to_csv(index=False).encode('utf-8'),
                       f'radar_{anio}.csv', 'text/csv')

# ───────────────────────────── Evolución ─────────────────────────────
with tab_evol:
    c1, c2 = st.columns([2, 1])
    top = radar.sort_values('residuo', ascending=False)['provincia'].head(3).tolist()
    elegidas = c1.multiselect('Provincias', sorted(radar['provincia']), default=top)
    desde, hasta = c2.slider('Período', 2000, 2025, (2010, 2025))
    serie = (df[df['delito'].isin(delitos) & df['provincia'].isin(elegidas) & df['anio'].between(desde, hasta)]
               .groupby(['provincia', 'anio'], as_index=False)['tasa_hechos'].sum())
    fig = px.line(serie, x='anio', y='tasa_hechos', color='provincia', markers=True,
                  color_discrete_sequence=PALETA, category_orders={'provincia': elegidas},
                  labels={'anio': '', 'tasa_hechos': 'Tasa cada 100.000 hab.', 'provincia': 'Provincia'})
    n = nac.loc[desde:hasta]
    fig.add_trace(go.Scatter(x=n.index, y=n.values, mode='lines', name='Argentina (total)',
                             line=dict(color='#0b0b0b', width=2, dash='dot')))
    if desde <= 2020 <= hasta:
        fig.add_vrect(x0=2019.5, x1=2020.5, fillcolor=GRIS, opacity=0.15, line_width=0,
                      annotation_text='Pandemia', annotation_position='top left')
    if desde <= 2019 and hasta >= 2017:
        fig.add_vrect(x0=2016.5, x1=2019.5, fillcolor=AZUL, opacity=0.06, line_width=0,
                      annotation_text='Prepandemia', annotation_position='top left')
    fig.update_layout(height=520, hovermode='x unified', margin=dict(t=30))
    st.plotly_chart(fig, width='stretch')
    st.caption('Por defecto se muestran las 3 provincias con mayor residuo. La línea punteada es la tasa nacional.')

# ───────────────────────────── Metodología ─────────────────────────────
with tab_metodo:
    st.markdown(f"""
#### Qué responde este tablero
¿En qué provincias la tasa de delitos contra la propiedad de **{anio}** quedó por encima de lo que su propio
nivel **prepandemia** permitía esperar, y cuáles ya superan ese nivel previo?

#### Cómo se calcula
1. **Datos:** SNIC por provincia, filtrado a hurtos, robos y robos agravados (1.872 filas, 2000-2025).
2. **Tasa:** hechos cada 100.000 habitantes; se suman las tasas de los delitos elegidos.
3. **x:** promedio 2017-2019 (nivel prepandemia). **y:** tasa del año elegido.
4. **Regresión lineal simple** por mínimos cuadrados entre las 24 provincias: ŷ = a + b·x.
5. **Residuo** = y − ŷ. Positivo = peor de lo esperado.
6. **Semáforo:** rojo si el residuo es positivo y además supera su nivel prepandemia; amarillo si solo el
   residuo es positivo; verde en el resto.

#### Limitaciones
- 24 observaciones: un caso extremo puede mover la recta (por ejemplo CABA o Neuquén).
- Es delito **registrado**: depende de las denuncias y de cómo registra cada provincia.
- Describe una asociación; no es una predicción ni explica causas.
""")
