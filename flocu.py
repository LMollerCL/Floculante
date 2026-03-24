import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Asistente de Dosificación IA", page_icon="⚙️", layout="wide")

st.title("⚙️ SISTEMA PREDICTIVO DE DOSIFICACIÓN DE REACTIVOS")
st.markdown("---")

# ==========================================
# 2. MOTOR DE IA (FUSIONADO Y CON FÍSICA)
# ==========================================
# Usamos cache para que solo entrene la primera vez y sea rápido
@st.cache_resource 
def entrenar_modelos():
    # Carga y limpieza (TK2601 + TK017)
    archivo = 'FloculanteLimpio.xlsx'
    df = pd.read_excel(archivo)
    
    # Procesar TK1
    df_tk1 = pd.DataFrame()
    df_tk1['TK_Hoy'] = df['TK2601']
    df_tk1['Tornillo_Hoy'] = df['Tornillo 6']
    df_tk1['Tornillo_Ayer'] = df['Tornillo 6'].shift(1)
    df_tk1['Tornillo_Hace_2'] = df['Tornillo 6'].shift(2)
    df_tk1['TK_24h'] = df['TK2601'].shift(-1)
    df_tk1['TK_48h'] = df['TK2601'].shift(-2)
    df_tk1 = df_tk1.dropna()
    
    # Procesar TK2
    df_tk2 = pd.DataFrame()
    df_tk2['TK_Hoy'] = df['TK017']
    df_tk2['Tornillo_Hoy'] = df['Tornillo 7']
    df_tk2['Tornillo_Ayer'] = df['Tornillo 7'].shift(1)
    df_tk2['Tornillo_Hace_2'] = df['Tornillo 7'].shift(2)
    df_tk2['TK_24h'] = df['TK017'].shift(-1)
    df_tk2['TK_48h'] = df['TK017'].shift(-2)
    df_tk2 = df_tk2.dropna()
    
    # Unir datos
    df_total = pd.concat([df_tk1, df_tk2], ignore_index=True)
    columnas_X = ['TK_Hoy', 'Tornillo_Hoy', 'Tornillo_Ayer', 'Tornillo_Hace_2']
    X = df_total[columnas_X].values
    y_24h = df_total['TK_24h'].values
    y_48h = df_total['TK_48h'].values
    
    # Entrenar IA (Bosques Aleatorios)
    modelo_24h = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    modelo_48h = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    modelo_24h.fit(X, y_24h)
    modelo_48h.fit(X, y_48h)
    
    return modelo_24h, modelo_48h

# Intentar cargar modelos
try:
    mod_24h, mod_48h = entrenar_modelos()
    st.success("🤖 Inteligencia Artificial entrenada y lista.")
except FileNotFoundError:
    st.error("❌ Error: No se encontró el archivo 'FloculanteLimpio.xlsx'. Asegúrate de que esté en la misma carpeta.")
    st.stop()

# ==========================================
# 3. INTERFAZ: BARRA LATERAL (ENTRADAS DE DATOS)
# ==========================================
st.sidebar.header("📥 INGRESO DE DATOS DEL TURNO")
st.sidebar.markdown("Complete los valores actuales para calcular la receta.")

# Sliders interactivos
tk_actual = st.sidebar.number_input("Nivel Actual del Estanque (TK):", value=3.50, min_value=1.0, max_value=8.0, step=0.01)
t_ayer = st.sidebar.slider("Minutos de Tornillo de AYER:", 0, 120, 60)
t_hace_2 = st.sidebar.slider("Minutos de Tornillo de HACE 2 DÍAS:", 0, 120, 60)

# Botón grande para calcular
boton_calcular = st.sidebar.button("⚙️ CALCULAR RECETA ÓPTIMA")

# ==========================================
# 4. LÓGICA DE CÁLCULO (HÍBRIDA CON FÍSICA)
# ==========================================
if boton_calcular:
    
    # --- Motor de Simulación y Decisión ---
    objetivo_ideal = 3.5
    tiempos_simulados = np.arange(0, 121, 1).reshape(-1, 1)
    n = len(tiempos_simulados)
    
    # Matrices para la IA
    tk_actual_arr = np.full((n, 1), tk_actual)
    ayer_arr = np.full((n, 1), t_ayer)
    hace_2_arr = np.full((n, 1), t_hace_2)
    
    X_escenarios = np.hstack((tk_actual_arr, tiempos_simulados, ayer_arr, hace_2_arr))
    
    # Predicciones crudas
    pred_brutas_48h = mod_48h.predict(X_escenarios)
    pred_brutas_24h = mod_24h.predict(X_escenarios)
    
    # --- FILTRO FÍSICO ANTI-ALUCINACIONES (HÍBRIDO) ---
    pred_cero_48h = pred_brutas_48h[0] 
    
    # Regla dura de Balance de Masa
    base_fisica_48h = pred_cero_48h
    if t_ayer == 0 and t_hace_2 == 0 and pred_cero_48h > tk_actual:
        base_fisica_48h = tk_actual # Forzamos la realidad
        
    # Extraer y sumar la Ganancia del Tornillo
    ganancia_tornillo_48h = pred_brutas_48h - pred_cero_48h
    pred_corregidas_48h = base_fisica_48h + ganancia_tornillo_48h
    
    # Decisión del mejor tiempo
    dif_48h = np.abs(pred_corregidas_48h - objetivo_ideal)
    indice_ganador = np.argmin(dif_48h)
    tiempo_rec = tiempos_simulados[indice_ganador][0]
    
    # Proyección Final Corregida (Línea de tiempo)
    # Día 2 (48h)
    t48 = pred_corregidas_48h[indice_ganador]
    
    # Día 1 (24h - Aplicando misma física)
    pred_cero_24h = pred_brutas_24h[0]
    base_fisica_24h = pred_cero_24h
    if t_hace_2 == 0 and pred_cero_24h > tk_actual:
        base_fisica_24h = tk_actual
    ganancia_tornillo_24h = pred_brutas_24h - pred_cero_24h
    t24 = base_fisica_24h + ganancia_tornillo_24h[indice_ganador]
    
    # Interpolaciones intermedias
    t12 = (tk_actual + t24) / 2
    t36 = (t24 + t48) / 2
    
    # ==========================================
    # 5. MOSTRAR RESULTADOS (EL DASHBOARD)
    # ==========================================
    st.header("📈 RESULTADOS DE LA PREDICCIÓN")
    
    # Fila 1: Indicadores clave (Metas y Estado)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="✅ META DE NIVEL (TK)", value="3.50", delta="Rango: 3.3 - 3.7")
        st.metric(label="👉 RECOMENDACIÓN HOY", value=f"{tiempo_rec} min.", delta="Tornillo 6")

    with col2:
        st.metric(label="📊 ESTADO ACTUAL (TK)", value=f"{tk_actual:.2f}")
        
    with col3:
        # Semáforo de Estado Final a 48h
        if 3.3 <= t48 <= 3.7:
            st.metric(label="🔮 ESTADO FINAL A 48H (Predicción)", value=f"{t48:.2f}", delta="✅ DENTRO DE RANGO", delta_color="normal")
        else:
            st.metric(label="🔮 ESTADO FINAL A 48H (Predicción)", value=f"{t48:.2f}", delta="⚠️ FUERA DE RANGO", delta_color="inverse")

    st.markdown("---")
    
    # Fila 2: Gráfico de Línea de Tiempo
    st.subheader("📋 Proyección del Estanque por Hora")
    
    horas = [0, 12, 24, 36, 48]
    niveles = [tk_actual, t12, t24, t36, t48]
    
    # Creación del gráfico con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(horas, niveles, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='Curva de Nivel')
    
    # Dibujar banda ideal (3.3 - 3.7)
    ax.axhspan(3.3, 3.7, color='green', alpha=0.15, label='Rango Ideal (3.3 - 3.7)')
    ax.axhline(3.5, color='green', linestyle='--', alpha=0.5, label='Meta central')
    
    # Etiquetas y Estilo
    ax.set_title('Evolución Proyectada del Nivel del TK2601')
    ax.set_xlabel('Horas desde el ajuste')
    ax.set_ylabel('Nivel del TK')
    ax.set_xticks(horas)
    ax.set_yticks([1, 2, 3, 3.3, 3.5, 3.7, 4, 5, 6, 7, 8])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')
    ax.set_ylim(min(niveles)-0.3, max(niveles)+0.3)
    
    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
    
    # Fila 3: Diagnóstico y Contingencia
    st.markdown("---")
    st.subheader("🛠️ Diagnóstico de Operación")
    
    if 3.3 <= t48 <= 3.7:
        st.success(f"La IA logró estabilizar el estanque en {t48:.2f}. Siga la recomendación del tornillo.")
    else:
        st.warning(f"Atención: El estanque quedará fuera del rango ideal (Proyección a 48h: {t48:.2f}).")
        
        # Lógica de Diagnóstico avanzado (el guardarraíl físico)
        if tiempo_rec == 0 and t48 > 3.7:
             st.info("🛑 DIAGNÓSTICO: La inercia del sistema es muy alta por los días pasados. El nivel subirá inevitablemente por el reactivo que ya viene en tránsito.")
             st.markdown("**🛠️ SUGERENCIA OPERATIVA:** Mantener el Tornillo 6 apagado (0 min). Monitorear si el nivel sigue subiendo más de lo proyectado y evaluar aumento de flujo de agua.")
        elif tiempo_rec == 120 and t48 < 3.3:
             st.info("🛑 DIAGNÓSTICO: Se alcanzó la capacidad máxima del Tornillo 6 (120 min) y no es suficiente para levantar el estanque.")
             st.markdown("**🛠️ SUGERENCIA OPERATIVA:** Revisar si hay taponamiento mecánico en el tornillo o baja densidad en la preparación del floculante.")

else:
    # Pantalla de bienvenida antes de calcular
    st.info("📥 Use la barra lateral para ingresar los datos del turno y presione 'Calcular Receta Óptima' para ver la proyección.")
    
    # Semáforo visual del rango
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Banda Operativa Ideal")
        st.success("🟢 3.3 - 3.7 (Nivel Óptimo)")
    with col2:
        st.markdown("### Alertas de Planta")
        st.warning("⚠️ < 3.3 o > 3.7 (Nivel Fuera de Rango)")