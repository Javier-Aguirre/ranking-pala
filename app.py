import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import math
from datetime import datetime
from openskill.models import PlackettLuce

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Liga Pala Pro", page_icon="⚾", layout="wide")
st.title("⚾ Liga Pala - Official Ranking")

conn = st.connection("gsheets", type=GSheetsConnection)
model = PlackettLuce(mu=25.0, sigma=8.333, beta=4.167, tau=0.083)

CATEGORY_PRIORS = {1: 30.0, 2: 27.5, 3: 25.0, 4: 22.5, 5: 20.0}

# ==========================================
# 2. GESTIÓN DE DATOS
# ==========================================

def load_data():
    # AHORA LEEMOS 7 COLUMNAS (Inc. parejas)
    # t1_j2 y t2_j2 pueden ser nulos (NaN) en partidos individuales
    df_partidos = conn.read(worksheet="Partidos", usecols=list(range(7)), ttl=0)
    df_partidos = df_partidos.fillna("") # Rellenar huecos con cadena vacía
    
    try:
        df_jugadores = conn.read(worksheet="Jugadores", usecols=[0, 1], ttl=0)
        if not df_jugadores.empty:
            df_jugadores['nombre'] = df_jugadores['nombre'].astype(str).str.strip().str.title()
    except:
        df_jugadores = pd.DataFrame(columns=['nombre', 'categoria'])
    return df_partidos, df_jugadores

def save_match(df_new):
    conn.update(worksheet="Partidos", data=df_new)

def save_new_players(df_old, new_list):
    if not new_list: return df_old
    new_df = pd.DataFrame(new_list)
    combined = pd.concat([df_old, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['nombre'])
    conn.update(worksheet="Jugadores", data=combined)
    return combined

# ==========================================
# 3. MOTOR DE CÁLCULO (SOPORTE PAREJAS)
# ==========================================

def compute_rankings(df_partidos, df_jugadores):
    ratings = {} 
    stats = {}
    elo_history = [] 
    
    cat_map = dict(zip(df_jugadores['nombre'], df_jugadores['categoria']))
    
    df_partidos['fecha'] = pd.to_datetime(df_partidos['fecha'])
    df_partidos = df_partidos.sort_values(by='fecha')

    for _, row in df_partidos.iterrows():
        # Extraer jugadores (limpiando vacíos)
        # Equipo 1
        team1_names = [str(row['t1_j1']).strip().title()]
        if row['t1_j2'] and str(row['t1_j2']).strip():
            team1_names.append(str(row['t1_j2']).strip().title())
            
        # Equipo 2
        team2_names = [str(row['t2_j1']).strip().title()]
        if row['t2_j2'] and str(row['t2_j2']).strip():
            team2_names.append(str(row['t2_j2']).strip().title())

        s1, s2 = row['puntos1'], row['puntos2']
        match_date = row['fecha']
        
        all_players = team1_names + team2_names
        
        # Inicialización de jugadores nuevos en el loop
        for p in all_players:
            if p not in ratings:
                cat = cat_map.get(p, 3)
                mu_init = CATEGORY_PRIORS.get(cat, 25.0)
                ratings[p] = model.rating(mu=mu_init, sigma=8.333, name=p)
                stats[p] = {'wins': 0, 'matches': 0}
                # Log inicial
                elo_start = int(((mu_init - 3 * 8.333) * 40) + 1000)
                elo_history.append({'Fecha': match_date, 'Jugador': p, 'ELO': elo_start})
            stats[p]['matches'] += 1

        # Construir objetos de equipo para Openskill
        # Openskill espera: [[p1_obj, p2_obj], [p3_obj, p4_obj]]
        team1_objs = [ratings[name] for name in team1_names]
        team2_objs = [ratings[name] for name in team2_names]
        
        # Determinar resultado
        if s1 > s2:
            ranks = [0, 1]; diff = s1 - s2
            for p in team1_names: stats[p]['wins'] += 1
        elif s2 > s1:
            ranks = [1, 0]; diff = s2 - s1
            for p in team2_names: stats[p]['wins'] += 1
        else:
            ranks = [0, 0]; diff = 0 # Empate

        # CÁLCULO BAYESIANO MULTIJUGADOR
        # model.rate acepta listas de longitud variable
        out = model.rate([team1_objs, team2_objs], ranks=ranks)
        
        mov_factor = max(0.5, min(math.log10(diff + 1) + 0.5, 2.0))

        # Función de update genérica
        def update_team(original_objs, new_objs, names):
            for i, name in enumerate(names):
                old = original_objs[i]
                new = new_objs[i]
                
                delta_mu = new.mu - old.mu
                delta_sigma = new.sigma - old.sigma
                
                final_mu = old.mu + (delta_mu * mov_factor)
                final_sigma = old.sigma + delta_sigma
                
                ratings[name] = model.create_rating([final_mu, final_sigma], name=name)
                
                # Log History
                current_elo = int(((final_mu - 3 * final_sigma) * 40) + 1000)
                elo_history.append({'Fecha': match_date, 'Jugador': name, 'ELO': current_elo})

        # Aplicar cambios
        update_team(team1_objs, out[0], team1_names)
        update_team(team2_objs, out[1], team2_names)

    # Generar DF Ranking
    data = []
    for name, r in ratings.items():
        conservative_elo = int(((r.mu - 3 * r.sigma) * 40) + 1000)
        confidence = max(0, min(100, (1 - (r.sigma / 8.333)) * 100))
        win_rate = (stats[name]['wins'] / stats[name]['matches']) if stats[name]['matches'] > 0 else 0
        
        data.append({
            'Jugador': name,
            'ELO': conservative_elo,
            'Cat': int(cat_map.get(name, 3)),
            'Win Rate': win_rate*100,
            'Confianza': confidence,
            'Partidos': stats[name]['matches']
        })
        
    ranking_df = pd.DataFrame(data)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(by='ELO', ascending=False).reset_index(drop=True)
        ranking_df.index += 1
        
    history_df = pd.DataFrame(elo_history)
    return ranking_df, history_df

# ==========================================
# 4. INTERFAZ GRÁFICA
# ==========================================

df_partidos, df_jugadores = load_data()
list_jugadores_existentes = sorted(df_jugadores['nombre'].unique().tolist()) if not df_jugadores.empty else []

if not df_partidos.empty:
    ranking_final, evolution_df = compute_rankings(df_partidos, df_jugadores)
else:
    ranking_final, evolution_df = pd.DataFrame(), pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Clasificación", "📈 Evolución", "📜 Historial", "🔒 Admin"])

with tab1: # RANKING (Igual que antes)
    if ranking_final.empty:
        st.info("Sin datos.")
    else:
        st.dataframe(ranking_final, width='stretch', column_config={
            "ELO": st.column_config.NumberColumn("ELO", format="%d ⚾"),
            "Win Rate": st.column_config.ProgressColumn(
            "Win Rate", 
                format="%d%%",   # Muestra "50%" en vez de "0%" o "1%"
                min_value=0, 
                max_value=100    # La escala ahora es hasta 100
            ),
            "Confianza": st.column_config.ProgressColumn(
                "Fiabilidad", 
                format="%d%%", 
                min_value=0, 
                max_value=100
            ),
            "Cat": st.column_config.NumberColumn("Cat.", format="%dª")
        })

with tab2: # GRAFICA (Igual que antes)
    if not evolution_df.empty:
        players = st.multiselect("Comparar:", ranking_final['Jugador'].tolist(), default=ranking_final['Jugador'].iloc[0])
        if players:
            st.line_chart(evolution_df[evolution_df['Jugador'].isin(players)], x='Fecha', y='ELO', color='Jugador')

with tab3: # HISTORIAL (Adaptado a parejas)
    st.header("Historial")
    if not df_partidos.empty:
        # Creamos una vista bonita para la tabla
        df_view = df_partidos.copy()
        
        # Función para formatear "Pepe" o "Pepe & Juan"
        def format_team(row, t_prefix):
            p1 = row[f'{t_prefix}_j1']
            p2 = row[f'{t_prefix}_j2']
            if p2: return f"{p1} & {p2}"
            return f"{p1}"

        df_view['Pareja 1'] = df_view.apply(lambda x: format_team(x, 't1'), axis=1)
        df_view['Pareja 2'] = df_view.apply(lambda x: format_team(x, 't2'), axis=1)
        df_view['Resultado'] = df_view.apply(lambda x: f"{int(x['puntos1'])} - {int(x['puntos2'])}", axis=1)
        
        st.dataframe(
            df_view[['fecha', 'Pareja 1', 'Resultado', 'Pareja 2']].sort_values(by='fecha', ascending=False),
            width='stretch',
            column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY")}
        )

with tab4: # ADMIN (FORMULARIO DINÁMICO)
    st.header("Nuevo Partido")
    password = st.text_input("Contraseña", type="password")
    
    if "admin" in st.secrets and password == st.secrets["admin"]["password"]:
        
        # SELECTOR DE MODO
        mode = st.radio("Tipo de Partido", ["Individual (1 vs 1)", "Parejas (2 vs 2)"], horizontal=True)
        is_doubles = mode == "Parejas (2 vs 2)"
        
        with st.form("match_form"):
            c_date = st.date_input("Fecha", datetime.now())
            st.divider()
            
            # --- LAYOUT DE EQUIPOS ---
            col_team1, col_score, col_team2 = st.columns([3, 1, 3])
            
            # Helper para crear inputs de jugador
            def player_input(label, key_suffix):
                sel = st.selectbox(f"Jugador {label}", ["- Nuevo -"] + list_jugadores_existentes, key=f"s_{key_suffix}")
                name, cat = None, None
                if sel == "- Nuevo -":
                    name = st.text_input(f"Nombre {label}", key=f"t_{key_suffix}").strip().title()
                    cat = st.selectbox(f"Cat {label}", [1,2,3,4,5], index=2, key=f"c_{key_suffix}")
                return sel, name, cat

            with col_team1:
                st.subheader("Pareja 1")
                # Siempre hay J1
                t1_p1_sel, t1_p1_new, t1_p1_cat = player_input("1", "t1p1")
                # Solo J2 si es dobles
                if is_doubles:
                    t1_p2_sel, t1_p2_new, t1_p2_cat = player_input("2 (Pareja)", "t1p2")
                else:
                    t1_p2_sel, t1_p2_new = None, None # placeholders

            with col_team2:
                st.subheader("Pareja 2")
                t2_p1_sel, t2_p1_new, t2_p1_cat = player_input("1", "t2p1")
                if is_doubles:
                    t2_p2_sel, t2_p2_new, t2_p2_cat = player_input("2 (Pareja)", "t2p2")
                else:
                    t2_p2_sel, t2_p2_new = None, None

            with col_score:
                st.write("") # Spacer
                st.write("") 
                s1 = st.number_input("Puntos E1", 0, 30, 0)
                s2 = st.number_input("Puntos E2", 0, 30, 0)

            submitted = st.form_submit_button("Registrar Resultado")
            
            if submitted:
                # 1. Resolver Nombres
                p_map = {} # Guardar nombre final y datos de nuevos
                
                # Lista de tuplas a procesar: (sel, new_name, new_cat, key_id)
                inputs_to_process = [
                    (t1_p1_sel, t1_p1_new, t1_p1_cat, 't1_j1'),
                    (t2_p1_sel, t2_p1_new, t2_p1_cat, 't2_j1')
                ]
                if is_doubles:
                    inputs_to_process += [
                        (t1_p2_sel, t1_p2_new, t1_p2_cat, 't1_j2'),
                        (t2_p2_sel, t2_p2_new, t2_p2_cat, 't2_j2')
                    ]
                
                new_players_batch = []
                final_names = {}
                error = False
                
                for sel, new_name, new_cat, key in inputs_to_process:
                    final_name = new_name if sel == "- Nuevo -" else sel
                    
                    if not final_name:
                        st.error(f"Falta nombre en {key}"); error = True
                    else:
                        final_names[key] = final_name
                        if sel == "- Nuevo -" and final_name not in list_jugadores_existentes:
                             new_players_batch.append({'nombre': final_name, 'categoria': new_cat})

                # Validar duplicados en el mismo partido
                all_names_in_match = list(final_names.values())
                if len(all_names_in_match) != len(set(all_names_in_match)):
                    st.error("Hay jugadores repetidos en el partido."); error = True
                
                if not error:
                    # Guardar nuevos jugadores
                    if new_players_batch:
                        save_new_players(df_jugadores, new_players_batch)
                    
                    # Preparar fila
                    row_data = {
                        "fecha": c_date.strftime("%Y-%m-%d"),
                        "t1_j1": final_names['t1_j1'],
                        "t1_j2": final_names.get('t1_j2', ""), # Vacío si es singles
                        "t2_j1": final_names['t2_j1'],
                        "t2_j2": final_names.get('t2_j2', ""),
                        "puntos1": s1,
                        "puntos2": s2
                    }
                    
                    # Guardar y Recargar
                    updated_matches = pd.concat([df_partidos, pd.DataFrame([row_data])], ignore_index=True)
                    save_match(updated_matches)
                    st.toast("Partido Registrado")
                    st.cache_data.clear()
                    st.rerun()

    elif password:
        st.error("Incorrecto")
