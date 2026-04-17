import re
import shutil
from typing import Dict, List, Tuple, Set, Optional

import pandas as pd
import streamlit as st
from graphviz import Digraph
from graphviz.backend import ExecutableNotFound

st.set_page_config(page_title="HTA Builder v7.2 Completa", layout="wide")

CODE_PATTERN = re.compile(r"^\d+(?:\.\d+)*$")

# =========================
# Helpers
# =========================
def natural_code_key(code: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(code).split("."))

def infer_parent(code: str) -> str:
    parts = str(code).split(".")
    return "" if len(parts) == 1 else ".".join(parts[:-1])

def level_from_code(code: str) -> int:
    return len(str(code).split(".")) - 1

def clean_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def wrap_lines(text: str, width: int = 26) -> str:
    words = str(text).split()
    if not words:
        return ""
    lines = []
    line = words[0]
    for word in words[1:]:
        if len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(lines)

def graphviz_available() -> bool:
    return shutil.which("dot") is not None

def safe_render(dot: Digraph, fmt: str = "svg"):
    try:
        return dot.pipe(format=fmt), None
    except ExecutableNotFound:
        return None, "No se encontró el ejecutable 'dot' de Graphviz en Windows. La visualización funciona, pero la exportación requiere instalar Graphviz y agregarlo al PATH."
    except Exception as e:
        return None, str(e)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# =========================
# Scoring dictionaries
# =========================
RIESGO_MAP = {"": 0, "muy bajo": 1, "bajo": 2, "medio": 3, "alto": 4, "crítico": 5, "critico": 5}
ESFUERZO_MAP = {"": 0, "muy bajo": 1, "bajo": 2, "moderado": 3, "alto": 4, "muy alto": 5}
FRECUENCIA_MAP = {"": 0, "única": 1, "ocasional": 2, "repetitiva": 3, "frecuente": 4, "constante": 5}
DURACION_MAP = {"": 0, "muy corta": 1, "corta": 2, "media": 3, "prolongada": 4}
ERR_PROB_MAP = {"": 0, "baja": 1, "media": 2, "alta": 3}
ERR_SEV_MAP = {"": 0, "leve": 1, "moderado": 2, "severo": 3, "crítico": 4, "critico": 4}

FACTOR_COLS = [
    "postura_forzada",
    "fuerza",
    "repeticion",
    "contacto_estres",
    "demanda_visual",
    "carga_mental",
    "ambiente_adverso",
]

BASE_COLS = [
    "code", "label", "parent", "plan", "type", "notes",
    "riesgo", "esfuerzo", "frecuencia", "duracion",
    "error", "consecuencia",
    "error_type", "error_description", "error_probability",
    "error_severity", "error_recovery",
] + FACTOR_COLS

# =========================
# Preprocess / validation
# =========================
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BASE_COLS:
        if col not in df.columns:
            df[col] = ""

    for col in BASE_COLS:
        df[col] = df[col].apply(clean_text)

    df["parent"] = df.apply(
        lambda r: r["parent"] if r["parent"] else infer_parent(r["code"]),
        axis=1
    )

    df["level"] = df["code"].apply(level_from_code)
    df = df.sort_values(by="code", key=lambda s: s.map(natural_code_key)).reset_index(drop=True)

    df["riesgo_score_manual"] = df["riesgo"].str.lower().map(RIESGO_MAP).fillna(0).astype(int)
    df["esfuerzo_score"] = df["esfuerzo"].str.lower().map(ESFUERZO_MAP).fillna(0).astype(int)
    df["frecuencia_score"] = df["frecuencia"].str.lower().map(FRECUENCIA_MAP).fillna(0).astype(int)
    df["duracion_score"] = df["duracion"].str.lower().map(DURACION_MAP).fillna(0).astype(int)
    df["error_prob_score"] = df["error_probability"].str.lower().map(ERR_PROB_MAP).fillna(0).astype(int)
    df["error_sev_score"] = df["error_severity"].str.lower().map(ERR_SEV_MAP).fillna(0).astype(int)

    factor_sum = []
    for _, row in df.iterrows():
        count = 0
        for c in FACTOR_COLS:
            if clean_text(row[c]).lower() == "sí":
                count += 1
        factor_sum.append(count)
    df["factores_presentes"] = factor_sum

    auto_raw = (
        0.35 * df["esfuerzo_score"] +
        0.25 * df["frecuencia_score"] +
        0.20 * df["duracion_score"] +
        0.20 * df["factores_presentes"]
    )
    df["riesgo_auto_score"] = auto_raw.round(2)

    def auto_label(x: float) -> str:
        if x <= 1.0:
            return "muy bajo"
        if x <= 2.0:
            return "bajo"
        if x <= 3.0:
            return "medio"
        if x <= 4.0:
            return "alto"
        return "crítico"

    df["riesgo_auto"] = df["riesgo_auto_score"].apply(auto_label)
    df["sherpa_priority"] = df["error_prob_score"] * df["error_sev_score"]
    df["sherpa_priority_adjusted"] = df["sherpa_priority"] + df["factores_presentes"]
    return df

def validate_dataframe(df: pd.DataFrame) -> List[str]:
    errors = []
    required = {"code", "label"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Faltan columnas requeridas: {', '.join(sorted(missing))}.")
        return errors

    seen = set()
    codes = set()

    for i, row in df.iterrows():
        code = clean_text(row["code"])
        label = clean_text(row["label"])

        if not code:
            errors.append(f"Fila {i+1}: el código está vacío.")
            continue
        if not CODE_PATTERN.match(code):
            errors.append(f"Fila {i+1}: código '{code}' inválido. Usa formato 0, 1, 1.1, 1.2.3.")
        if code in seen:
            errors.append(f"Fila {i+1}: código duplicado '{code}'.")
        seen.add(code)

        if not label:
            errors.append(f"Fila {i+1}: la descripción de '{code}' está vacía.")
        codes.add(code)

    for i, row in df.iterrows():
        code = clean_text(row["code"])
        parent = clean_text(row.get("parent", "")) or infer_parent(code)
        if parent and parent not in codes:
            errors.append(f"Fila {i+1}: el padre '{parent}' de '{code}' no existe en la tabla.")
    return errors

# =========================
# Tree utilities
# =========================
def build_children_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        children.setdefault(row["parent"], []).append(row["code"])
    for k in children:
        children[k] = sorted(children[k], key=natural_code_key)
    return children

def descendants(root: str, children_map: Dict[str, List[str]]) -> Set[str]:
    result = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for child in children_map.get(cur, []):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result

def subtree_df(df: pd.DataFrame, root_code: str) -> pd.DataFrame:
    cmap = build_children_map(df)
    keep = descendants(root_code, cmap)
    out = df[df["code"].isin(keep)].copy()
    return out.sort_values(by="code", key=lambda s: s.map(natural_code_key)).reset_index(drop=True)

def filter_to_max_level(df: pd.DataFrame, max_level: Optional[int]) -> pd.DataFrame:
    if max_level is None:
        return df.copy()
    return df[df["level"] <= max_level].copy().reset_index(drop=True)

# =========================
# Styling / graph
# =========================
def node_fill(row: pd.Series, color_mode: str) -> str:
    risk_map = {
        "muy bajo": "#e5e7eb",
        "bajo": "#dcfce7",
        "medio": "#fef3c7",
        "alto": "#fed7aa",
        "crítico": "#fecaca",
        "critico": "#fecaca",
    }
    sherpa_map = {
        "leve": "#e5e7eb",
        "moderado": "#fde68a",
        "severo": "#fdba74",
        "crítico": "#fecaca",
        "critico": "#fecaca",
    }
    type_map = {
        "cognitiva": "#dbeafe",
        "motora": "#dcfce7",
        "perceptiva": "#fef3c7",
        "ambiental": "#f3e8ff",
        "decisión": "#fee2e2",
        "decision": "#fee2e2",
    }
    level_map = {
        0: "#f3f4f6",
        1: "#e5f3ff",
        2: "#eefce8",
        3: "#fff4df",
        4: "#f5ecff",
        5: "#ffe8ef",
    }
    factor_map = {
        0: "#ffffff",
        1: "#e0f2fe",
        2: "#fef3c7",
        3: "#fed7aa",
        4: "#fecaca",
        5: "#fecaca",
        6: "#fecaca",
        7: "#fecaca",
    }

    if color_mode == "Riesgo automático":
        return risk_map.get(clean_text(row["riesgo_auto"]).lower(), "#ffffff")
    if color_mode == "Riesgo manual":
        return risk_map.get(clean_text(row["riesgo"]).lower(), "#ffffff")
    if color_mode == "SHERPA":
        return sherpa_map.get(clean_text(row["error_severity"]).lower(), "#ffffff")
    if color_mode == "Tipo de tarea":
        return type_map.get(clean_text(row["type"]).lower(), "#ffffff")
    if color_mode == "Factores ergonómicos":
        return factor_map.get(int(row["factores_presentes"]), "#ffffff")
    return level_map.get(int(row["level"]), "#ffffff")

def make_node_label(row: pd.Series, show_metrics: bool, show_sherpa: bool, show_factors: bool, show_notes: bool, wrap_width: int) -> str:
    lines = [str(row["code"]), wrap_lines(row["label"], wrap_width)]

    if show_metrics:
        metric_parts = []
        if row["riesgo"]:
            metric_parts.append(f'Riesgo manual: {row["riesgo"]}')
        if row["riesgo_auto"]:
            metric_parts.append(f'Riesgo auto: {row["riesgo_auto"]}')
        if row["esfuerzo"]:
            metric_parts.append(f'Esfuerzo: {row["esfuerzo"]}')
        if row["frecuencia"]:
            metric_parts.append(f'Freq.: {row["frecuencia"]}')
        if row["duracion"]:
            metric_parts.append(f'Dur.: {row["duracion"]}')
        if metric_parts:
            lines.append("—")
            lines.extend([wrap_lines(x, wrap_width) for x in metric_parts])

    if show_sherpa:
        sherpa_parts = []
        if row["error_type"]:
            sherpa_parts.append(f'Tipo error: {row["error_type"]}')
        if row["error_probability"]:
            sherpa_parts.append(f'Prob.: {row["error_probability"]}')
        if row["error_severity"]:
            sherpa_parts.append(f'Sev.: {row["error_severity"]}')
        if row["sherpa_priority_adjusted"] > 0:
            sherpa_parts.append(f'Prioridad: {int(row["sherpa_priority_adjusted"])}')
        if sherpa_parts:
            lines.append("—")
            lines.extend([wrap_lines(x, wrap_width) for x in sherpa_parts])

    if show_factors:
        factors = [c for c in FACTOR_COLS if clean_text(row[c]).lower() == "sí"]
        if factors:
            lines.append("—")
            lines.append(f"Factores: {len(factors)}")

    if show_notes and clean_text(row["notes"]):
        lines.append("—")
        lines.append(wrap_lines(f'Obs.: {row["notes"]}', wrap_width))

    return "\n".join([x for x in lines if x])

def build_hta_graph(
    df: pd.DataFrame,
    color_mode: str,
    show_metrics: bool,
    show_sherpa: bool,
    show_factors: bool,
    show_notes: bool,
    show_plans: bool,
    font_size: int = 15,
    plan_font_size: int = 11,
    wrap_width: int = 24,
    node_margin: str = "0.14,0.10",
    nodesep: float = 0.35,
    ranksep: float = 0.55,
) -> Digraph:
    dot = Digraph("HTA")
    dot.attr(
        rankdir="TB",
        splines="polyline",
        bgcolor="white",
        nodesep=str(nodesep),
        ranksep=str(ranksep),
        pad="0.25",
        margin="0.05",
    )
    dot.attr("node", shape="box", style="rounded,filled", color="black", penwidth="1.4", fontname="Arial")
    dot.attr("edge", color="black", penwidth="1.2", arrowsize="0.0")

    available_codes = set(df["code"])

    for _, row in df.iterrows():
        label = make_node_label(row, show_metrics, show_sherpa, show_factors, show_notes, wrap_width)
        dot.node(
            row["code"],
            label=label,
            fillcolor=node_fill(row, color_mode),
            fontsize=str(font_size),
            margin=node_margin,
        )

    for _, row in df.iterrows():
        if row["parent"] and row["parent"] in available_codes:
            dot.edge(row["parent"], row["code"])

    parent_groups: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        parent_groups.setdefault(row["parent"], []).append(row["code"])

    for parent, children in parent_groups.items():
        if parent and len(children) > 1:
            with dot.subgraph() as s:
                s.attr(rank="same")
                for child in sorted(children, key=natural_code_key):
                    s.node(child)

    if show_plans:
        for _, row in df.iterrows():
            if row["plan"]:
                plan_id = f'plan_{row["code"].replace(".", "_")}'
                plan_text = f'Plan {row["code"]}\n{wrap_lines(row["plan"], 28)}'
                dot.node(
                    plan_id,
                    label=plan_text,
                    shape="note",
                    style="filled",
                    fillcolor="#ffffff",
                    color="#666666",
                    penwidth="0.8",
                    fontsize=str(plan_font_size),
                    fontname="Arial",
                )
                dot.edge(row["code"], plan_id, style="invis", weight="25")
                with dot.subgraph() as s:
                    s.attr(rank="same")
                    s.node(row["code"])
                    s.node(plan_id)
    return dot

# =========================
# Example data
# =========================
def example_dataframe() -> pd.DataFrame:
    rows = [
        ["0", "Ir al baño en hospital sin asistencia", "", "hacer 1, 2, 3, 4 y 5 en secuencia", "", "Objetivo principal del análisis", "medio", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["1", "Decidir ir al baño", "0", "hacer 1.1, 1.2, 1.3 y 1.4", "decisión", "Proceso inicial de valoración", "medio", "moderado", "ocasional", "corta", "Decisión tardía", "Demora o acción insegura", "decisión", "Decide actuar sin ayuda", "media", "moderado", "sí", "no", "no", "no", "no", "no", "sí", "no"],
        ["1.1", "Percibir necesidad fisiológica", "1", "", "perceptiva", "Puede alterarse por medicación o fatiga", "bajo", "bajo", "ocasional", "corta", "", "", "información", "No reconoce adecuadamente la necesidad", "baja", "leve", "sí", "no", "no", "no", "no", "sí", "no", "no"],
        ["1.2", "Evaluar urgencia", "1", "", "cognitiva", "Depende de interpretación del malestar", "medio", "moderado", "ocasional", "corta", "", "", "comprobación", "Evalúa mal la urgencia", "media", "moderado", "sí", "no", "no", "no", "no", "sí", "sí", "no"],
        ["1.3", "Evaluar capacidad para movilizarse", "1", "", "cognitiva", "Tarea crítica por posible sobreestimación", "alto", "moderado", "ocasional", "corta", "Subestimar limitaciones", "Riesgo de caída", "comprobación", "Sobreestima su capacidad física", "alta", "severo", "no", "no", "no", "no", "no", "no", "sí", "sí"],
        ["1.4", "Decidir actuar sin asistencia", "1", "", "decisión", "Puede estar influida por urgencia y contexto", "alto", "moderado", "ocasional", "corta", "", "", "selección", "Elige una opción insegura", "media", "severo", "parcial", "no", "no", "no", "no", "no", "sí", "sí"],
        ["2", "Salir de la cama sin ayuda del staff", "0", "hacer 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 y 2.7", "motora", "Bloque con mayor exigencia física", "alto", "alto", "ocasional", "media", "Transferencia inestable", "Caída o sobrecarga", "", "", "", "", "", "sí", "sí", "no", "no", "no", "no", "sí"],
        ["2.1", "Ajustar posición en la cama", "2", "", "motora", "Movimiento preparatorio", "medio", "moderado", "ocasional", "corta", "", "", "acción", "Movimiento ineficaz", "media", "moderado", "sí", "sí", "sí", "no", "no", "no", "no", "no"],
        ["2.2", "Girar el cuerpo hacia el borde", "2", "", "motora", "Requiere coordinación y control", "medio", "moderado", "ocasional", "corta", "", "", "acción", "Giro descontrolado", "media", "moderado", "sí", "sí", "sí", "no", "no", "no", "no", "no"],
        ["2.3", "Desplazar piernas fuera de la cama", "2", "", "motora", "Transición hacia sedente", "medio", "moderado", "ocasional", "corta", "", "", "acción", "Secuencia motora deficiente", "media", "moderado", "sí", "sí", "no", "no", "no", "no", "no", "no"],
        ["2.4", "Incorporar el tronco para sentarse", "2", "", "motora", "Alta exigencia de control postural", "alto", "alto", "ocasional", "corta", "", "", "acción", "Incorporación inestable", "alta", "severo", "parcial", "sí", "sí", "no", "no", "no", "no", "no"],
        ["2.5", "Estabilizar postura sedente", "2", "", "motora", "Debe lograrse antes de ponerse de pie", "alto", "moderado", "ocasional", "corta", "", "", "comprobación", "No verifica estabilidad suficiente", "media", "severo", "parcial", "sí", "no", "no", "no", "no", "no", "no"],
        ["2.6", "Transferirse a bipedestación", "2", "", "motora", "Punto de máximo riesgo", "crítico", "muy alto", "ocasional", "corta", "Pérdida de equilibrio", "Caída", "acción", "Pérdida de equilibrio al ponerse de pie", "alta", "crítico", "no", "sí", "sí", "no", "no", "no", "no", "sí"],
        ["2.7", "Mantener equilibrio inicial", "2", "", "motora", "Determinante para continuar marcha", "alto", "alto", "ocasional", "corta", "", "", "acción", "Balance insuficiente", "alta", "severo", "parcial", "sí", "no", "no", "no", "no", "no", "no"],
        ["3", "Prepararse para caminar", "0", "hacer 3.1, 3.2, 3.3 y 3.4", "motora", "Bloque de transición funcional", "alto", "alto", "frecuente", "media", "", "", "", "", "", "", "", "sí", "sí", "sí", "no", "no", "no", "sí"],
        ["3.1", "Evaluar estabilidad corporal", "3", "", "cognitiva", "Verificación previa al desplazamiento", "medio", "moderado", "ocasional", "corta", "", "", "comprobación", "No identifica inestabilidad", "media", "moderado", "sí", "no", "no", "no", "no", "no", "sí", "no"],
        ["3.2", "Identificar apoyos cercanos", "3", "", "ambiental", "Depende del entorno disponible", "medio", "bajo", "ocasional", "corta", "", "", "información", "No localiza apoyo disponible", "media", "moderado", "sí", "no", "no", "no", "no", "sí", "no", "sí"],
        ["3.3", "Ajustar postura", "3", "", "motora", "Preparación para inicio de marcha", "medio", "moderado", "frecuente", "corta", "", "", "acción", "Postura inicial inadecuada", "media", "moderado", "sí", "sí", "no", "sí", "no", "no", "no", "no"],
        ["3.4", "Iniciar marcha", "3", "", "motora", "Inicio puede ser inseguro", "alto", "alto", "frecuente", "media", "", "", "acción", "Inicio de marcha sin control", "alta", "severo", "parcial", "sí", "sí", "sí", "no", "no", "no", "sí"],
    ]
    return pd.DataFrame(rows, columns=BASE_COLS)

# =========================
# UI
# =========================
st.title("HTA Builder v7.2 Completa")
st.caption("Versión integrada: análisis de v7 Pro + mejoras visuales + notas visibles + modo paper")

with st.sidebar:
    st.header("Módulos")
    usar_sherpa = st.toggle("Activar SHERPA", value=True)
    usar_factores = st.toggle("Activar factores ergonómicos", value=True)

    st.header("Visualización")
    color_mode = st.selectbox(
        "Colorear nodos por",
        ["Riesgo automático", "Riesgo manual", "SHERPA", "Tipo de tarea", "Factores ergonómicos", "Nivel"],
        index=0,
    )
    show_metrics = st.toggle("Mostrar métricas ergonómicas en nodos", value=True)
    show_sherpa = st.toggle("Mostrar métricas SHERPA en nodos", value=True)
    show_factors = st.toggle("Mostrar conteo de factores en nodos", value=True)
    show_notes = st.toggle("Mostrar notas/observaciones en nodos", value=True)
    show_plans = st.toggle("Mostrar planes", value=True)
    view_mode = st.radio("Modo de visualización", ["HTA completo", "Dividir por tarea principal"], index=1)
    max_level_option = st.selectbox("Mostrar hasta nivel", ["Todos", "0", "1", "2", "3", "4", "5", "6"], index=4)

    st.header("Ajuste visual")
    font_size = st.slider("Tamaño de texto del nodo", 8, 24, 14)
    plan_font_size = st.slider("Tamaño de texto del plan", 8, 20, 11)
    wrap_width = st.slider("Ancho de texto por línea", 16, 40, 24)
    margin_x = st.slider("Padding horizontal del nodo", 8, 30, 14)
    margin_y = st.slider("Padding vertical del nodo", 6, 20, 10)
    nodesep = st.slider("Separación horizontal entre nodos", 0.2, 1.5, 0.35, 0.05)
    ranksep = st.slider("Separación vertical entre niveles", 0.2, 1.8, 0.55, 0.05)

    st.header("Salida")
    modo_paper = st.toggle("Modo paper", value=True)

max_level = None if max_level_option == "Todos" else int(max_level_option)
node_margin = f"{margin_x/100:.2f},{margin_y/100:.2f}"

st.markdown(
    "**Columnas mínimas:** `code`, `label`  \n"
    "**Base:** `plan`, `type`, `notes`, `riesgo`, `esfuerzo`, `frecuencia`, `duracion`, `error`, `consecuencia`  \n"
    "**SHERPA:** `error_type`, `error_description`, `error_probability`, `error_severity`, `error_recovery`  \n"
    "**Factores:** `postura_forzada`, `fuerza`, `repeticion`, `contacto_estres`, `demanda_visual`, `carga_mental`, `ambiente_adverso`"
)

tab1, tab2, tab3 = st.tabs(["Editor manual", "Subir archivo", "Plantilla"])

with tab1:
    df0 = example_dataframe()

    visible_cols = ["code", "label", "parent", "plan", "type", "notes", "riesgo", "esfuerzo", "frecuencia", "duracion", "error", "consecuencia"]
    if usar_sherpa:
        visible_cols += ["error_type", "error_description", "error_probability", "error_severity", "error_recovery"]
    if usar_factores:
        visible_cols += FACTOR_COLS

    editor_df = df0[visible_cols].copy()

    column_config = {
        "code": st.column_config.TextColumn("code", help="Ej.: 0, 1, 1.1, 1.1.1"),
        "label": st.column_config.TextColumn("label", width="large"),
        "parent": st.column_config.TextColumn("parent"),
        "plan": st.column_config.TextColumn("plan", width="large"),
        "type": st.column_config.SelectboxColumn("type", options=["", "motora", "cognitiva", "perceptiva", "ambiental", "decisión"]),
        "notes": st.column_config.TextColumn("notes", width="large"),
        "riesgo": st.column_config.SelectboxColumn("riesgo", options=["", "muy bajo", "bajo", "medio", "alto", "crítico"]),
        "esfuerzo": st.column_config.SelectboxColumn("esfuerzo", options=["", "muy bajo", "bajo", "moderado", "alto", "muy alto"]),
        "frecuencia": st.column_config.SelectboxColumn("frecuencia", options=["", "única", "ocasional", "repetitiva", "frecuente", "constante"]),
        "duracion": st.column_config.SelectboxColumn("duracion", options=["", "muy corta", "corta", "media", "prolongada"]),
        "error": st.column_config.TextColumn("error", width="large"),
        "consecuencia": st.column_config.TextColumn("consecuencia", width="large"),
    }

    if usar_sherpa:
        column_config.update({
            "error_type": st.column_config.SelectboxColumn("error_type", options=["", "acción", "comprobación", "selección", "información", "búsqueda", "decisión", "omisión"]),
            "error_description": st.column_config.TextColumn("error_description", width="large"),
            "error_probability": st.column_config.SelectboxColumn("error_probability", options=["", "baja", "media", "alta"]),
            "error_severity": st.column_config.SelectboxColumn("error_severity", options=["", "leve", "moderado", "severo", "crítico"]),
            "error_recovery": st.column_config.SelectboxColumn("error_recovery", options=["", "sí", "parcial", "no"]),
        })

    if usar_factores:
        for c in FACTOR_COLS:
            column_config[c] = st.column_config.SelectboxColumn(c, options=["", "sí", "no"])

    edited_df = st.data_editor(
        editor_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        key="editor_v72",
    )

with tab2:
    uploaded = st.file_uploader("Sube un archivo .csv o .xlsx", type=["csv", "xlsx"])
    upload_df = None
    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            upload_df = pd.read_csv(uploaded)
        else:
            upload_df = pd.read_excel(uploaded)
        st.dataframe(upload_df, use_container_width=True)

with tab3:
    tmpl = example_dataframe()
    st.dataframe(tmpl.head(10), use_container_width=True)
    st.download_button("Descargar plantilla CSV", data=to_csv_bytes(tmpl), file_name="hta_template_v7_2_completa.csv", mime="text/csv")

source_df = upload_df if "upload_df" in locals() and upload_df is not None else edited_df

if st.button("Generar HTA v7.2", type="primary", use_container_width=True):
    df = preprocess_df(source_df)
    errors = validate_dataframe(df)

    if errors:
        st.error("Corrige estos problemas antes de generar el HTA:")
        for err in errors:
            st.write(f"- {err}")
    else:
        filtered_df = filter_to_max_level(df, max_level)

        if not usar_sherpa:
            filtered_df["error_type"] = ""
            filtered_df["error_description"] = ""
            filtered_df["error_probability"] = ""
            filtered_df["error_severity"] = ""
            filtered_df["error_recovery"] = ""
            filtered_df["error_prob_score"] = 0
            filtered_df["error_sev_score"] = 0
            filtered_df["sherpa_priority"] = 0
            filtered_df["sherpa_priority_adjusted"] = 0
            show_sherpa = False

        if not usar_factores:
            for c in FACTOR_COLS:
                filtered_df[c] = ""
            filtered_df["factores_presentes"] = 0
            show_factors = False

        st.success("HTA generado.")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tareas", len(filtered_df))
        c2.metric("Riesgo auto alto/crítico", int(filtered_df["riesgo_auto"].isin(["alto", "crítico"]).sum()))
        c3.metric("Tareas con factores", int((filtered_df["factores_presentes"] > 0).sum()))
        c4.metric("Errores SHERPA severo/crítico", int((filtered_df["error_sev_score"] >= 3).sum()) if usar_sherpa else 0)
        c5.metric("Prioridad SHERPA alta", int((filtered_df["sherpa_priority_adjusted"] >= 9).sum()) if usar_sherpa else 0)

        st.subheader("Matriz maestra")
        matrix_cols = ["code", "label", "parent", "type", "notes", "riesgo", "riesgo_auto", "esfuerzo", "frecuencia", "duracion", "factores_presentes", "error", "consecuencia"]
        if usar_sherpa:
            matrix_cols += ["error_type", "error_description", "error_probability", "error_severity", "error_recovery", "sherpa_priority_adjusted"]
        st.dataframe(filtered_df[matrix_cols], use_container_width=True, hide_index=True)

        top_level = sorted(filtered_df[filtered_df["parent"] == "0"]["code"].tolist(), key=natural_code_key)

        if view_mode == "HTA completo":
            dot = build_hta_graph(
                filtered_df,
                color_mode=color_mode,
                show_metrics=show_metrics,
                show_sherpa=show_sherpa and usar_sherpa,
                show_factors=show_factors and usar_factores,
                show_notes=show_notes,
                show_plans=show_plans,
                font_size=font_size,
                plan_font_size=plan_font_size,
                wrap_width=wrap_width,
                node_margin=node_margin,
                nodesep=nodesep,
                ranksep=ranksep,
            )
            st.subheader("Vista HTA")
            st.graphviz_chart(dot, use_container_width=True)

            b1, b2, b3 = st.columns(3)
            b1.download_button("Descargar DOT", dot.source, "hta_v7_2_completa.dot", "text/plain", use_container_width=True)
            svg_bytes, svg_error = safe_render(dot, "svg")
            if svg_bytes is not None:
                b2.download_button("Descargar SVG", svg_bytes, "hta_v7_2_completa.svg", "image/svg+xml", use_container_width=True)
            else:
                b2.button("SVG no disponible", disabled=True, use_container_width=True)
            b3.download_button("Descargar CSV", to_csv_bytes(filtered_df), "hta_v7_2_completa.csv", "text/csv", use_container_width=True)

            if not graphviz_available():
                st.warning("No tienes Graphviz instalado en Windows o no está en PATH. Puedes visualizar el HTA, pero no exportarlo como imagen.")
            elif svg_error:
                st.warning(svg_error)

        else:
            st.subheader("Vista dividida por tarea principal")
            if not top_level:
                st.info("No se encontraron tareas principales hijas de 0 con el filtro actual.")
            for code in top_level:
                label = filtered_df.loc[filtered_df["code"] == code, "label"].iloc[0]
                st.markdown(f"## {code} - {label}")
                part_df = subtree_df(filtered_df, code)
                dot = build_hta_graph(
                    part_df,
                    color_mode=color_mode,
                    show_metrics=show_metrics,
                    show_sherpa=show_sherpa and usar_sherpa,
                    show_factors=show_factors and usar_factores,
                    show_notes=show_notes,
                    show_plans=show_plans,
                    font_size=font_size,
                    plan_font_size=plan_font_size,
                    wrap_width=wrap_width,
                    node_margin=node_margin,
                    nodesep=nodesep,
                    ranksep=ranksep,
                )
                st.graphviz_chart(dot, use_container_width=True)
                st.dataframe(part_df[matrix_cols], use_container_width=True, hide_index=True)

        st.subheader("Tareas críticas priorizadas")
        priority_df = filtered_df[["code", "label", "riesgo_auto", "riesgo_auto_score", "factores_presentes"]].copy()
        if usar_sherpa:
            priority_df["sherpa_prioridad"] = filtered_df["sherpa_priority_adjusted"]
        else:
            priority_df["sherpa_prioridad"] = 0

        def prioridad_intervencion(row):
            score = row["riesgo_auto_score"] + row["factores_presentes"] + row["sherpa_prioridad"]
            if score >= 12:
                return "intervención inmediata"
            if score >= 8:
                return "intervención prioritaria"
            if score >= 5:
                return "seguimiento"
            return "baja prioridad"

        priority_df["prioridad_intervencion"] = priority_df.apply(prioridad_intervencion, axis=1)
        order_map = {
            "intervención inmediata": 1,
            "intervención prioritaria": 2,
            "seguimiento": 3,
            "baja prioridad": 4,
        }
        priority_df["orden"] = priority_df["prioridad_intervencion"].map(order_map)
        priority_df = priority_df.sort_values(by=["orden", "riesgo_auto_score", "sherpa_prioridad"], ascending=[True, False, False]).drop(columns=["orden"])
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

        if modo_paper:
            st.subheader("Tablas para paper")

            table_hta = filtered_df[["code", "label", "parent", "plan", "type", "notes"]].copy()
            st.markdown("**Tabla 1. Estructura HTA**")
            st.dataframe(table_hta, use_container_width=True, hide_index=True)

            table_ergo = filtered_df[["code", "label", "riesgo", "riesgo_auto", "esfuerzo", "frecuencia", "duracion", "factores_presentes", "error", "consecuencia", "notes"]].copy()
            st.markdown("**Tabla 2. Matriz ergonómica**")
            st.dataframe(table_ergo, use_container_width=True, hide_index=True)

            paper_export = {
                "hta_table": table_hta,
                "ergo_table": table_ergo,
            }

            if usar_sherpa:
                table_sherpa = filtered_df[[
                    "code", "label", "error_type", "error_description",
                    "error_probability", "error_severity", "error_recovery",
                    "sherpa_priority_adjusted", "notes"
                ]].copy()
                st.markdown("**Tabla 3. Matriz SHERPA**")
                st.dataframe(table_sherpa, use_container_width=True, hide_index=True)
                paper_export["sherpa_table"] = table_sherpa

            if usar_factores:
                table_factores = filtered_df[["code", "label"] + FACTOR_COLS + ["factores_presentes", "notes"]].copy()
                st.markdown("**Tabla 4. Factores de riesgo ergonómico**")
                st.dataframe(table_factores, use_container_width=True, hide_index=True)
                paper_export["factores_table"] = table_factores

            paper_csv = pd.concat(
                [df.assign(tabla=name) for name, df in paper_export.items()],
                ignore_index=True,
                sort=False,
            ).to_csv(index=False).encode("utf-8")

            st.download_button(
                "Descargar tablas paper (CSV unificado)",
                data=paper_csv,
                file_name="hta_v7_2_tablas_paper.csv",
                mime="text/csv",
                use_container_width=True,
            )
else:
    st.info("Edita la tabla o sube un archivo, luego presiona 'Generar HTA v7.2'.")
