# app.py
# ================================================================
# "Veterano - Pagante (DELTAS #2) | 5 Dispersões + 2 Heatmaps + Barras (IC95%)"
# Streamlit + Plotly — usa DELTAS em pp (D_CONV, D_REND100, …) e AUM em %
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import atanh, tanh, sqrt
from scipy.stats import pearsonr, spearmanr, norm

st.set_page_config(page_title="Veterano - Pagante (DELTAS #2)", layout="wide")

ROXO = "#9900FF"
NUM_COLS = ["D_CONV","D_REND100","D_MENOR40","D_REPROV","D_NAO_AI","D_MIX_INAD","AUM","D_DEV_BOLSA"]
ALL_COLS = ["MARCA"] + NUM_COLS

st.title("Veterano - Pagante (DELTAS #2) • 5 Dispersões + 2 Heatmaps + Barras (IC95%)")
st.caption("Interativo (Streamlit + Plotly). Upload de CSV opcional; parser aceita vírgula decimal e sufixos %/pp.")

# ---------------------------
# Helpers
# ---------------------------
def to_float(x):
    """Converte '1,2 pp' ou '35,5%' → float (1.2 / 35.5)."""
    if pd.isna(x) or x == "":
        return np.nan
    s = str(x).lower().replace("%","").replace("pp","").replace(",",".").strip()
    try:
        return float(s)
    except:
        return np.nan

def fisher_ci(r, n, alpha=0.05):
    """IC 95% do r de Pearson (Fisher)."""
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = atanh(r); se = 1.0 / sqrt(n - 3); zcrit = norm.ppf(1 - alpha/2.0)
    return tanh(z - zcrit*se), tanh(z + zcrit*se)

def add_quadrant_lines(fig, x0, x1, y0, y1):
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)
    fig.add_vline(x=0, line_dash="dash", opacity=0.6)
    fig.update_xaxes(range=[x0, x1])
    fig.update_yaxes(range=[y0, y1])

def scatter_corr(df, y_col, y_label, title):
    x = df["D_CONV"]; y = df[y_col]
    dx = (x.max() - x.min())*0.1 if np.isfinite(x.max()-x.min()) else 1
    dy = (y.max() - y.min())*0.1 if np.isfinite(y.max()-y.min()) else 1
    x0, x1 = x.min()-dx, x.max()+dx
    y0, y1 = y.min()-dy, y.max()+dy

    fig = px.scatter(
        df, x="D_CONV", y=y_col, text="MARCA",
        color="AUM", color_continuous_scale="Viridis_r",
        labels={"D_CONV":"Δ % Conversão vs AA (pp)", y_col:y_label, "AUM":"Δ % AUM Percebido"},
        title=title
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=0.6, color="black")),
                      textposition="top center")
    add_quadrant_lines(fig, x0, x1, y0, y1)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    return fig

def heatmap_corr(df, method="pearson"):
    corr = df[NUM_COLS].corr(method=method)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        colorbar=dict(title="Correlação")
    ))
    # anotações por célula
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            fig.add_annotation(
                x=col, y=row, text=f"{corr.iloc[i,j]:.2f}",
                showarrow=False, font=dict(size=12, color="black"),
                xanchor="center", yanchor="middle",
                bgcolor="rgba(255,255,255,0.7)"
            )
    fig.update_layout(
        title=f"Heatmap de Correlação — {method.capitalize()} (Deltas)",
        xaxis=dict(tickangle=45),
        margin=dict(l=80,r=20,t=60,b=120)
    )
    return fig

def barras_correlacoes(df):
    var_map = {
        "D_REND100": "REND",
        "D_MENOR40": "Média <40",
        "D_REPROV" : "Reprovação",
        "D_NAO_AI" : "NAO_AI",
        "D_MIX_INAD":"Inadimplência",
        "AUM"      : "AUM"
    }
    x_var = "D_CONV"
    labels, rvals, lo_list, hi_list = [], [], [], []
    for col, lbl in var_map.items():
        sub = df[[x_var, col]].dropna(); n = len(sub)
        if n < 3:
            r, lo, hi = np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(sub[x_var], sub[col]); lo, hi = fisher_ci(r, n)
        labels.append(lbl); rvals.append(r); lo_list.append(lo); hi_list.append(hi)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rvals,
        marker_color=ROXO, marker_line=dict(color="black", width=0.8),
        name="Correlação (Pearson)"
    ))
    y_err_plus  = [ (hi - r) if np.isfinite(hi) and np.isfinite(r) else 0 for r, hi in zip(rvals, hi_list) ]
    y_err_minus = [ (r - lo) if np.isfinite(lo) and np.isfinite(r) else 0 for r, lo in zip(rvals, lo_list) ]
    fig.update_traces(error_y=dict(
        type='data', symmetric=False, array=y_err_plus, arrayminus=y_err_minus,
        thickness=1.2, width=5, color="black"
    ))
    fig.add_hline(y=0, line_dash="dash", opacity=0.7)
    fig.update_layout(
        title="Correlação com Δ % Conversão (IC 95%)",
        yaxis_title="Coeficiente de Correlação (Pearson)",
        margin=dict(l=40,r=20,t=60,b=40)
    )
    return fig

# ---------------------------
# Base padrão (se não subir CSV)
# ---------------------------
raw_default = [
 ["ÂNIMA BR","1,0 pp","0,5 pp","-0,3 pp","-0,9 pp","-17,5 pp","-4,1 pp","35,5%","-18,3 pp"],
 ["AGES","1,0 pp","-0,3 pp","-1,2 pp","0,0 pp","-18,9 pp","-2,2 pp","119,0%","-22,4 pp"],
 ["UNIFG - BA","1,5 pp","1,0 pp","-1,1 pp","-1,6 pp","-25,2 pp","-8,8 pp","102,5%","-26,5 pp"],
 ["UNF","1,2 pp","0,6 pp","-0,5 pp","-0,9 pp","-20,0 pp","-4,9 pp","98,6%","-18,4 pp"],
 ["UNP","1,5 pp","0,3 pp","-0,1 pp","-0,8 pp","-14,5 pp","-4,8 pp","95,2%","-13,5 pp"],
 ["FPB","0,3 pp","0,0 pp","-0,7 pp","-1,0 pp","-4,9 pp","-5,7 pp","70,8%","-7,8 pp"],
 ["UNIFG - PE","2,1 pp","1,3 pp","-0,8 pp","-2,2 pp","-20,3 pp","-4,7 pp","93,5%","-14,6 pp"],
 ["UAM","1,0 pp","-1,8 pp","0,9 pp","1,4 pp","-19,3 pp","-2,6 pp","16,4%","0,0 pp"],
 ["USJT","0,8 pp","-1,0 pp","0,0 pp","0,4 pp","-16,5 pp","-2,2 pp","23,7%","0,0 pp"],
 ["UNA","2,2 pp","0,1 pp","-0,1 pp","-0,7 pp","-12,8 pp","-6,2 pp","25,3%","0,0 pp"],
 ["UNIBH","-0,7 pp","-0,9 pp","0,2 pp","0,7 pp","-27,0 pp","-4,1 pp","21,0%","0,0 pp"],
 ["IBMR","-0,3 pp","6,2 pp","-1,7 pp","-6,3 pp","-31,6 pp","-2,2 pp","95,0%","-22,2 pp"],
 ["FASEH","3,1 pp","-0,2 pp","-0,3 pp","0,0 pp","-4,8 pp","-9,3 pp","46,1%","0,0 pp"],
 ["MIL. CAMPOS","-0,1 pp","-2,0 pp","-2,2 pp","1,7 pp","0,0 pp","0,6 pp","4,1%","0,0 pp"],
 ["UNISUL","0,5 pp","0,9 pp","-0,6 pp","-0,6 pp","-29,9 pp","-2,4 pp","14,4%","0,0 pp"],
 ["UNICURITIBA","-1,1 pp","-0,3 pp","0,2 pp","-0,1 pp","-23,3 pp","-3,6 pp","3,6%","0,0 pp"],
 ["UNISOCIESC","0,3 pp","1,4 pp","-0,3 pp","-2,2 pp","-16,1 pp","-1,4 pp","19,4%","0,0 pp"],
 ["UNR","0,3 pp","2,0 pp","-0,3 pp","-1,7 pp","-13,0 pp","-9,4 pp","117,3%","-23,7 pp"],
 ["FAD","2,9 pp","3,8 pp","-1,8 pp","-3,9 pp","-33,7 pp","-11,0 pp","94,5%","-16,1 pp"]
]
df_default = pd.DataFrame(raw_default, columns=ALL_COLS)
for c in NUM_COLS:
    df_default[c] = df_default[c].apply(to_float)

# ---------------------------
# Sidebar: Upload CSV (opcional)
# ---------------------------
st.sidebar.header("Dados")
st.sidebar.write("CSV esperado com colunas:")
st.sidebar.code(", ".join(ALL_COLS), language="text")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded, dtype=str)
        missing = [c for c in ALL_COLS if c not in df.columns]
        if missing:
            st.sidebar.error(f"Faltam colunas no CSV: {missing}. Usando base de exemplo.")
            df = df_default.copy()
        else:
            for c in NUM_COLS:
                df[c] = df[c].apply(to_float)
    except Exception as e:
        st.sidebar.error(f"Erro ao ler CSV: {e}. Usando base de exemplo.")
        df = df_default.copy()
else:
    df = df_default.copy()

# ---------------------------
# Abas e gráficos
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Dispersões", "Heatmaps", "Barras (IC95%)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            scatter_corr(df, "D_MENOR40", "Δ % Média < 40 (pp)", "Δ Conversão (X) vs Δ % Média < 40 (Y)"),
            use_container_width=True
        )
    with c2:
        st.plotly_chart(
            scatter_corr(df, "D_REPROV", "Δ % Reprovado (pp)", "Δ Conversão (X) vs Δ % Reprovado (Y)"),
            use_container_width=True
        )
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            scatter_corr(df, "D_REND100", "Δ % Rendimento 100% (pp)", "Δ Conversão (X) vs Δ % Rendimento 100% (Y)"),
            use_container_width=True
        )
    with c4:
        st.plotly_chart(
            scatter_corr(df, "D_MIX_INAD", "Δ % Mix Inadimplência (pp)", "Δ Conversão (X) vs Δ % Mix Inadimplência (Y)"),
            use_container_width=True
        )
    st.plotly_chart(
        scatter_corr(df, "D_NAO_AI", "Δ % Não Realizou AI (pp)", "Δ Conversão (X) vs Δ % Não Realizou AI (Y)"),
        use_container_width=True
    )

with tab2:
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(heatmap_corr(df, method="pearson"), use_container_width=True)
    with colB:
        st.plotly_chart(heatmap_corr(df, method="spearman"), use_container_width=True)

with tab3:
    st.plotly_chart(barras_correlacoes(df), use_container_width=True)

st.markdown("---")
st.caption(f"Barras em roxo {ROXO}. Dispersões com escala Viridis invertida (cor = Δ % AUM).")
