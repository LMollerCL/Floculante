import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. CARGA Y ENTRENAMIENTO (Se hace una sola vez al abrir)
# ==========================================
print("Cargando historial y entrenando Inteligencia Artificial... Por favor espere.")

archivo = 'FloculanteLimpio.xlsx'
df = pd.read_excel(archivo)

# Procesar TK2601
df_tk1 = pd.DataFrame()
df_tk1['TK_Hoy'] = df['TK2601']
df_tk1['Tornillo_Hoy'] = df['Tornillo 6']
df_tk1['Tornillo_Ayer'] = df['Tornillo 6'].shift(1)
df_tk1['Tornillo_Hace_2'] = df['Tornillo 6'].shift(2)
df_tk1['TK_24h'] = df['TK2601'].shift(-1)
df_tk1['TK_48h'] = df['TK2601'].shift(-2)
df_tk1 = df_tk1.dropna()

# Procesar TK017
df_tk2 = pd.DataFrame()
df_tk2['TK_Hoy'] = df['TK017']
df_tk2['Tornillo_Hoy'] = df['Tornillo 7']
df_tk2['Tornillo_Ayer'] = df['Tornillo 7'].shift(1)
df_tk2['Tornillo_Hace_2'] = df['Tornillo 7'].shift(2)
df_tk2['TK_24h'] = df['TK017'].shift(-1)
df_tk2['TK_48h'] = df['TK017'].shift(-2)
df_tk2 = df_tk2.dropna()

# Unir y preparar datos
df_total = pd.concat([df_tk1, df_tk2], ignore_index=True)
columnas_X = ['TK_Hoy', 'Tornillo_Hoy', 'Tornillo_Ayer', 'Tornillo_Hace_2']
X = df_total[columnas_X].values
y_24h = df_total['TK_24h'].values
y_48h = df_total['TK_48h'].values

# Entrenar modelos
modelo_24h = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
modelo_48h = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
modelo_24h.fit(X, y_24h)
modelo_48h.fit(X, y_48h)

# ==========================================
# 2. EL MOTOR PREDICTIVO (CON FÍSICA APLICADA)
# ==========================================
def proyeccion_tk_con_memoria(tk_actual, tornillo_ayer, tornillo_hace_2):
    objetivo_ideal = 3.5  
    
    # 1. Simular todos los escenarios posibles (0 a 120 min)
    tiempos_posibles = np.arange(0, 121, 1).reshape(-1, 1)
    n = len(tiempos_posibles)
    
    tk_actual_arr = np.full((n, 1), tk_actual)
    ayer_arr = np.full((n, 1), tornillo_ayer)
    hace_2_arr = np.full((n, 1), tornillo_hace_2)
    
    escenarios_X = np.hstack((tk_actual_arr, tiempos_posibles, ayer_arr, hace_2_arr))
    
    # Predicciones "crudas" de la IA
    predicciones_brutas_48h = modelo_48h.predict(escenarios_X)
    predicciones_brutas_24h = modelo_24h.predict(escenarios_X)
    
    # --- 2. FILTRO FÍSICO ANTI-ALUCINACIONES ---
    # ¿Qué dice la IA que pasa si el tornillo HOY es 0?
    pred_cero_48h = predicciones_brutas_48h[0] 
    
    # REGLA DE MASA: Si no hay reactivo de ayer, ni de hace 2 días, y hoy es 0...
    # el estanque NO PUEDE SUBIR. Físicamente solo puede bajar o mantenerse.
    base_fisica_48h = pred_cero_48h
    if tornillo_ayer == 0 and tornillo_hace_2 == 0 and pred_cero_48h > tk_actual:
        base_fisica_48h = tk_actual  # Forzamos la realidad: la base es donde estamos
        
    # Extraemos la "Ganancia Pura" que la IA sabe que da el tornillo
    ganancia_tornillo_48h = predicciones_brutas_48h - pred_cero_48h
    
    # Sumamos la ganancia real a nuestra base física corregida
    predicciones_corregidas_48h = base_fisica_48h + ganancia_tornillo_48h
    
    # --- 3. DECISIÓN OPERATIVA ---
    # Ahora sí, buscamos qué tiempo nos deja más cerca de 3.5 usando los datos corregidos
    diferencias = np.abs(predicciones_corregidas_48h - objetivo_ideal)
    mejor_indice = np.argmin(diferencias)
    tiempo_recomendado = tiempos_posibles[mejor_indice][0]
    
    # 4. Calcular la línea de tiempo usando el índice ganador
    t48 = predicciones_corregidas_48h[mejor_indice]
    
    # Aplicar la misma física para el día 1 (24h)
    pred_cero_24h = predicciones_brutas_24h[0]
    base_fisica_24h = pred_cero_24h
    if tornillo_hace_2 == 0 and pred_cero_24h > tk_actual:
        base_fisica_24h = tk_actual
        
    ganancia_tornillo_24h = predicciones_brutas_24h - pred_cero_24h
    predicciones_corregidas_24h = base_fisica_24h + ganancia_tornillo_24h
    t24 = predicciones_corregidas_24h[mejor_indice]
    
    # Interpolar para visualización
    t12 = (tk_actual + t24) / 2
    t36 = (t24 + t48) / 2
    
    return tiempo_recomendado, t12, t24, t36, t48

# ==========================================
# 3. INTERFAZ INTERACTIVA PARA EL OPERADOR
# ==========================================
# Limite máximo de tu planta (ajústalo si es distinto a 120)
MAX_MINUTOS_TORNILLO = 120 

while True:
    print("\n" + "="*75)
    print("⚙️  SISTEMA PREDICTIVO DE DOSIFICACIÓN DE REACTIVOS ⚙️")
    print("    Meta Operativa: Mantener el nivel entre 3.3 y 3.7")
    print("="*75)
    
    try:
        # Solicitud de datos al operador
        tk_hoy = float(input("▶️ Ingrese el NIVEL ACTUAL del estanque (ej. 3.2): "))
        t_ayer = float(input("▶️ Ingrese los MINUTOS de tornillo de AYER (ej. 60): "))
        t_hace_2 = float(input("▶️ Ingrese los MINUTOS de tornillo de HACE 2 DÍAS (ej. 45): "))
        
        print("\n⏳ Calculando la ruta óptima considerando la inercia del sistema...")
        
        tiempo, t12, t24, t36, t48 = proyeccion_tk_con_memoria(tk_hoy, t_ayer, t_hace_2)
        
        print("\n" + "-"*75)
        print(f"👉 ACCIÓN RECOMENDADA HOY: Encender el tornillo por {tiempo} minutos.")
        print("-" * 75)
        print("📈 PROYECCIÓN DE IMPACTO (Próximas 48 horas):")
        print(f"   [ +00 hrs ] Nivel: {tk_hoy:.2f} (Ahora)")
        print(f"   [ +12 hrs ] Nivel: {t12:.2f}")
        print(f"   [ +24 hrs ] Nivel: {t24:.2f} (Impacto histórico en tránsito)")
        print(f"   [ +36 hrs ] Nivel: {t36:.2f}")
        print(f"   [ +48 hrs ] Nivel: {t48:.2f} (Estado final esperado)")
        
        # --- NUEVA LÓGICA DE DIAGNÓSTICO Y ALARMAS ---
        if 3.3 <= t48 <= 3.7:
            print(f"\n   ✅ RESULTADO A 48H: Excelente. El estanque se estabilizará en {t48:.2f} (Dentro de norma).")
        else:
            print(f"\n   ⚠️ ALERTA: El estanque quedará fuera del rango ideal (Proyección: {t48:.2f}).")
            
            # Explicación del motivo y contingencia operativa
            if tiempo == 0:
                print("   🛑 DIAGNÓSTICO: La inercia del sistema es muy alta. El reactivo de los días pasados")
                print("                   empujará el nivel hacia arriba inevitablemente.")
                print("   🛠️ SUGERENCIA:  Mantener tornillo apagado (0 min). Monitorear y evaluar aumento")
                print("                   de flujo de extracción o dilución si el nivel sigue subiendo. Volver a realizar este diagnóstico mañana")
            elif tiempo == MAX_MINUTOS_TORNILLO:
                print(f"   🛑 DIAGNÓSTICO: Se alcanzó la capacidad máxima del tornillo ({MAX_MINUTOS_TORNILLO} min).")
                print("                   No es suficiente para levantar el nivel del estanque al rango 3.3.")
                print("   🛠️ SUGERENCIA:  Revisar si hay taponamiento mecánico en el tornillo o caída en la")
                print("                   densidad de la preparación.")
            else:
                print("   🛑 DIAGNÓSTICO: Complejidad física del proceso. Con los parámetros actuales, ningún")
                print("                   tiempo de tornillo asegura entrar exactamente en la banda 3.3 - 3.7.")
        print("="*75)
        
    except ValueError:
        print("\n❌ Error: Por favor, ingrese solo valores numéricos. Use punto (.) para los decimales.")
    
    continuar = input("\n¿Desea realizar un nuevo cálculo? (s/n): ").strip().lower()
    if continuar != 's':
        print("\nCerrando el sistema predictivo. ¡Buen turno!")
        break