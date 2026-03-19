#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "streamlit",
#   "sqlalchemy",
#   "psycopg2-binary",
#   "python-dotenv",
#   "pandas",
#   "plotly"
# ]
# ///
import json
import os
from datetime import date

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Layout Blocks Explorer", layout="wide")

def refresh_data() -> None:
    load_from_db.clear()
    st.cache_data.clear()
    
# -----------------------
# Env / DB config
# -----------------------
load_dotenv()  # loads .env in current working dir
DB_URL = os.getenv("DATABASE_URL", "").strip()
MINIO_PUBLIC_BASE = "https://minio.smartbiblia.fr"

if not DB_URL:
    st.error("DB_URL not found. Add it to your .env file, e.g. DB_URL=postgresql+psycopg2://user:pass@host:5432/db")
    st.stop()

# Optional: set schema name here if not "vlm_eval"
DB_SCHEMA = os.getenv("DB_SCHEMA", "vlm_eval").strip()

# -----------------------
# Data loader
# -----------------------
@st.cache_data(show_spinner=False)
def load_from_db(db_url: str, schema: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Requires:
      pip install sqlalchemy psycopg2-binary python-dotenv

    Expects these objects (adapt schema if needed):
      {schema}.layout_annotations_norm
      {schema}.block_types
      {schema}.cases
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)

    q_annotations = text(f"""
        SELECT
          -- stringify IDs to avoid UUID not JSON serializable
          CAST(lan.layout_annotation_id AS text) AS layout_annotation_id,
          CAST(lan.campaign_id AS text)          AS campaign_id,
          CAST(lan.case_id AS text)              AS case_id,
          CAST(lan.block_type_id AS text)        AS block_type_id,

          lan.x1, lan.y1, lan.x2, lan.y2,
          lan.w, lan.h,
          lan.x1n, lan.y1n, lan.x2n, lan.y2n,
          lan.cxn, lan.cyn,

          bt.code  AS block_code,
          bt.label AS block_label,

          c.doc_type,
          c.memoire_type_code,
          c.year,
          c.collection_code,
          c.is_humatheque

        FROM {schema}.layout_annotations_norm lan

        JOIN {schema}.block_types bt
          ON bt.block_type_id = CAST(lan.block_type_id AS uuid)

        JOIN {schema}.cases c
          ON c.case_id = CAST(lan.case_id AS uuid)
    """)
    q_cases = text(f"""
        SELECT
          CAST(c.case_id AS text) AS case_id,
          c.case_name,
          c.doc_type,
          c.memoire_type_code,
          c.year,
          c.collection_code,
          c.is_humatheque,
          c.source_ref,
          c.created_at
        FROM {schema}.cases c
    """)
    with engine.begin() as cx:
        df = pd.read_sql(q_annotations, cx)
        cases_df = pd.read_sql(q_cases, cx)

    # typing
    for col in ["x1n", "y1n", "x2n", "y2n", "cxn", "cyn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "year" in cases_df.columns:
        cases_df["year"] = pd.to_numeric(cases_df["year"], errors="coerce")
    if "is_humatheque" in df.columns:
        # keep boolean, but tolerate strings
        if df["is_humatheque"].dtype == object:
            df["is_humatheque"] = df["is_humatheque"].astype(str).str.lower().map({"true": True, "false": False})
    if "is_humatheque" in cases_df.columns and cases_df["is_humatheque"].dtype == object:
        cases_df["is_humatheque"] = cases_df["is_humatheque"].astype(str).str.lower().map({"true": True, "false": False})
    return df, cases_df

def ensure_bbox(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bw"] = (df["x2n"] - df["x1n"]).clip(lower=0)
    df["bh"] = (df["y2n"] - df["y1n"]).clip(lower=0)
    df["barea"] = (df["bw"] * df["bh"]).clip(lower=0)
    return df

df, cases_df = load_from_db(DB_URL, DB_SCHEMA)
df = ensure_bbox(df)

# Canonical block display label
df["block"] = df["block_code"].fillna(df["block_type_id"].astype(str))
df["block_display"] = df["block_label"].fillna(df["block"]).astype(str)

# -----------------------
# Sidebar filters (facets)
# -----------------------
st.sidebar.title("Filtres")

# Block filter
block_options = (
    df[["block", "block_display"]]
    .drop_duplicates()
    .sort_values("block_display")
)
block_display_list = block_options["block_display"].tolist()
default_blocks = block_display_list  # select all by default

sel_block_display = st.sidebar.multiselect("Blocs", block_display_list, default=default_blocks)
sel_block_codes = set(block_options[block_options["block_display"].isin(sel_block_display)]["block"].tolist())

mask = df["block"].isin(sel_block_codes)
cases_mask = pd.Series(True, index=cases_df.index)

# doc_type facet
doc_types = sorted([x for x in cases_df["doc_type"].dropna().unique().tolist()])
if doc_types:
    sel_doc_types = st.sidebar.multiselect("doc_type", doc_types, default=doc_types)
    mask &= df["doc_type"].isin(sel_doc_types)
    if "doc_type" in cases_df.columns:
        cases_mask &= cases_df["doc_type"].isin(sel_doc_types)

# memoire_type_code facet
memoire_types = sorted([x for x in cases_df["memoire_type_code"].dropna().unique().tolist()])
if cases_df["memoire_type_code"].isna().any():
    memoire_types = memoire_types + ["(null)"]
if memoire_types:
    sel_memoire_types = st.sidebar.multiselect("memoire_type_code", memoire_types, default=memoire_types)
    memoire_mask = df["memoire_type_code"].isin([x for x in sel_memoire_types if x != "(null)"])
    if "(null)" in sel_memoire_types:
        memoire_mask |= df["memoire_type_code"].isna()
    mask &= memoire_mask
    cases_memoire_mask = cases_df["memoire_type_code"].isin([x for x in sel_memoire_types if x != "(null)"])
    if "(null)" in sel_memoire_types:
        cases_memoire_mask |= cases_df["memoire_type_code"].isna()
    cases_mask &= cases_memoire_mask

# is_humatheque facet
if cases_df["is_humatheque"].notna().any():
    sel_huma = st.sidebar.selectbox("is_humatheque", ["(tous)", True, False], index=0)
    if sel_huma != "(tous)":
        mask &= (df["is_humatheque"] == sel_huma)
        cases_mask &= (cases_df["is_humatheque"] == sel_huma)

# collection_code facet
collections = sorted([x for x in cases_df["collection_code"].dropna().unique().tolist()])
if collections:
    sel_cols = st.sidebar.multiselect("collection_code", collections, default=collections)
    mask &= df["collection_code"].isin(sel_cols)
    if "collection_code" in cases_df.columns:
        cases_mask &= cases_df["collection_code"].isin(sel_cols)

# year facet (range)
if cases_df["year"].notna().any():
    y_min = int(np.nanmin(cases_df["year"]))
    y_max = int(np.nanmax(cases_df["year"]))
    yr = st.sidebar.slider("Années", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    mask &= df["year"].between(yr[0], yr[1])
    cases_mask &= cases_df["year"].between(yr[0], yr[1])

dff = df.loc[mask].copy()
cases_ff = cases_df.loc[cases_mask].copy()
cases_ff["image_url"] = cases_ff["source_ref"].apply(
    lambda s: f"{MINIO_PUBLIC_BASE}/{str(s).lstrip('/')}" if pd.notna(s) and str(s).strip() else None
)

# -----------------------
# Header KPIs
# -----------------------
title_col, refresh_col = st.columns([6, 1])
with title_col:
    st.title("Layout Blocks Explorer — Statistiques sur blocs annotés")
with refresh_col:
    st.write("")
    if st.button("Refresh", use_container_width=True, help="Recharge les données depuis la base et met à jour tous les compteurs."):
        refresh_data()
        st.rerun()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annotations (lignes)", f"{len(dff):,}".replace(",", " "))
c2.metric("Docs (case_id)", f"{cases_ff['case_id'].nunique():,}".replace(",", " "))
c3.metric("Types de blocs", f"{dff['block'].nunique():,}".replace(",", " "))
c4.metric("Campagnes", f"{dff['campaign_id'].nunique():,}".replace(",", " "))

st.divider()

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Carte des positions",
    "Distributions",
    "Qualité / outliers",
    "Synthèse 'zones probables'",
    "Couverture / données manquantes",
    "Recherche"
])

# ---- Tab 1: positions map
with tab1:
    st.subheader("Nuage de centroïdes (0..1) + densité")
    with st.expander("Comment lire ces graphiques ?"):
        st.markdown(
        """
    Cette vue montre **où les blocs se situent sur la page** après normalisation des coordonnées entre 0 et 1.

    **Nuage de points**
    → chaque point correspond au **centre d’un bloc annoté** dans un document.

    **Couleur**
    → chaque couleur représente un **type de bloc**.

    **Axes normalisés**
    → `x = 0` est à gauche, `x = 1` à droite ; `y = 0` est en haut de la page et `y = 1` en bas.

    **Ce qu’il faut regarder**
    - un amas serré indique une position plutôt stable
    - un nuage large indique une position plus variable
    - plusieurs amas distincts peuvent révéler plusieurs mises en page

    **Heatmap**
    → elle résume la concentration des centroïdes pour un bloc choisi.

    **Interprétation**
    - zones foncées = positions fréquentes
    - zones claires ou diffuses = positions rares ou dispersées
    - plusieurs zones chaudes = comportements différents selon les documents
        """
        )

    colA, colB = st.columns([2, 1])

    with colA:
        fig = px.scatter(
            dff,
            x="cxn", y="cyn",
            color="block_display",
            opacity=0.55,
            hover_data=["case_id", "doc_type", "year", "collection_code", "is_humatheque"],
            title="Centroïdes normalisés (par bloc)",
        )
        fig.update_yaxes(autorange="reversed")  # origin at top like a page
        st.plotly_chart(fig, width="stretch")

    with colB:
        one_block_display = st.selectbox("Bloc pour heatmap", block_display_list, index=0)
        one_block_code = block_options.loc[block_options["block_display"] == one_block_display, "block"].iloc[0]
        dd = dff[dff["block"] == one_block_code]
        if len(dd) < 5:
            st.info("Pas assez de points pour une heatmap.")
        else:
            hfig = px.density_heatmap(
                dd, x="cxn", y="cyn",
                nbinsx=25, nbinsy=25,
                title=f"Heatmap des centroïdes — {one_block_display}"
            )
            hfig.update_yaxes(autorange="reversed")
            st.plotly_chart(hfig, width="stretch")

# ---- Tab 2: distributions
with tab2:
    st.subheader("Variabilité des positions et tailles des blocs")
    with st.expander("Comment lire ces boxplots ?"):
        st.markdown(
        """
    Ces graphiques montrent **la distribution des positions des blocs annotés** sur la page. Chaque point correspond à un **bloc annoté dans un document**, et les boxplots permettent de résumer cette distribution.

    ### Éléments du boxplot

    **Médiane (ligne dans la boîte)**  
    → valeur centrale de la distribution.  
    50 % des blocs sont au-dessus et 50 % en dessous.

    **Q1 (premier quartile)**  
    → 25 % des observations sont en dessous.

    **Q3 (troisième quartile)**  
    → 75 % des observations sont en dessous.

    **Boîte (entre Q1 et Q3)**  
    → contient les **50 % centraux des observations**.

    **IQR (Interquartile Range)**  
    → IQR = Q3 − Q1  
    → mesure la **dispersion principale** de la distribution.

    ### Whiskers (moustaches)

    Les moustaches s'étendent jusqu'à :

    - **Lower fence** = Q1 − 1.5 × IQR  
    - **Upper fence** = Q3 + 1.5 × IQR  

    Les valeurs au-delà sont considérées comme **outliers**.

    ### Points (outliers)

    Les points visibles hors des moustaches représentent :

    → des **positions atypiques** du bloc dans certains documents.

    Cela peut indiquer :

    - pages de titre atypiques  
    - documents atypiques   
    - variations de structure

    ### Interprétation générale

    Un bloc est **positionnellement stable** lorsque :

    - la **boîte est étroite** (position stable)
    - il y a **peu d’outliers**
    - la médiane est clairement identifiable

    À l’inverse, une distribution **large ou très dispersée** indique que :

    → la position du bloc varie fortement selon les documents.
        """
        )

    col1, col2 = st.columns(2)

    with col1:
        figy = px.violin(
            dff, x="block_display", y="cyn", box=True, points="outliers",
            title="Distribution verticale (cyn) par bloc"
        )
        figy.update_yaxes(autorange="reversed")
        st.plotly_chart(figy, width="stretch")

    with col2:
        figx = px.violin(
            dff, x="block_display", y="cxn", box=True, points="outliers",
            title="Distribution horizontale (cxn) par bloc"
        )
        st.plotly_chart(figx, width="stretch")

    st.caption("Lecture rapide : un bloc 'stable' apparaît avec une distribution serrée et peu de points extrêmes.")

    st.subheader("Comparaisons (facettes)")
    colA, colB = st.columns(2)

    with colA:
        comp_block_display = st.selectbox("Bloc à comparer", block_display_list, index=0, key="comp_block")
        comp_block_code = block_options.loc[block_options["block_display"] == comp_block_display, "block"].iloc[0]
        dd = dff[dff["block"] == comp_block_code]
        if dd["doc_type"].nunique() >= 2:
            fig = px.box(dd, x="doc_type", y="cyn", points="outliers",
                         title=f"{comp_block_display} — position verticale selon doc_type")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Pas assez de diversité doc_type pour comparer.")

    with colB:
        dd = dff[dff["block"] == comp_block_code]
        if dd["collection_code"].nunique() >= 2:
            # avoid too many categories
            top_cols = dd["collection_code"].value_counts().head(20).index.tolist()
            ddd = dd[dd["collection_code"].isin(top_cols)]
            fig = px.box(ddd, x="collection_code", y="cyn", points=False,
                         title=f"{comp_block_display} — cyn selon collection_code (top 20)")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Pas assez de diversité collection_code pour comparer.")

# ---- Tab 3: outliers
with tab3:
    st.subheader("Détection de documents atypiques (outliers)")
    with st.expander("Comment lire ces graphiques ?"):
        st.markdown(
        """
    Cette vue met en évidence les **annotations qui s’écartent fortement** de la position moyenne de leur bloc.

    **Distance normalisée `z`**
    → plus `z` est élevé, plus la position observée est éloignée du comportement habituel du bloc.

    **Carte d’un bloc**
    → les points gris représentent les annotations ordinaires, les points rouges les annotations atypiques.
    Le marqueur noir indique le **centre moyen** du bloc.

    **Barres par bloc**
    → elles classent les blocs selon la **part d’annotations atypiques**.
    La couleur signale l’intensité maximale observée.

    **Nuage par document**
    → chaque point représente un document.
    Plus il est à droite, plus le document contient de blocs atypiques ; plus il est haut, plus l’écart maximal est fort.

    **Interprétation**
    - quelques points rouges isolés = anomalies ponctuelles
    - beaucoup de rouge pour un bloc = bloc instable ou cas hétérogènes
    - documents très à droite et très hauts = cas particulièrement atypiques
        """
        )

    stats = dff.groupby("block")[["cxn", "cyn"]].agg(["mean", "std"]).reset_index()
    stats.columns = ["block", "cx_mean", "cx_std", "cy_mean", "cy_std"]

    dd = dff.merge(stats, on="block", how="left")
    dd["cx_std"] = dd["cx_std"].replace(0, np.nan)
    dd["cy_std"] = dd["cy_std"].replace(0, np.nan)
    dd["z"] = np.sqrt(((dd["cxn"] - dd["cx_mean"]) / dd["cx_std"])**2 + ((dd["cyn"] - dd["cy_mean"]) / dd["cy_std"])**2)
    dd["z"] = dd["z"].replace([np.inf, -np.inf], np.nan)

    zthr = st.slider("Seuil z (distance normalisée)", 1.0, 8.0, 3.0, 0.1)
    out = dd[dd["z"] >= zthr].copy()

    total_annotations = len(dd)
    outlier_docs = out["case_id"].nunique()
    per_block = (
        dd.groupby("block_display")
        .agg(
            n_annotations=("case_id", "size"),
            mean_z=("z", "mean"),
        )
        .reset_index()
    )
    per_block_out = (
        out.groupby("block_display")
        .agg(
            n_outliers=("case_id", "size"),
            max_z=("z", "max"),
        )
        .reset_index()
    )
    per_block = per_block.merge(per_block_out, on="block_display", how="left")
    per_block["n_outliers"] = per_block["n_outliers"].fillna(0)
    per_block["max_z"] = per_block["max_z"].fillna(0)
    per_block["outlier_rate"] = per_block["n_outliers"] / per_block["n_annotations"].replace(0, np.nan)
    per_block = per_block.sort_values(["outlier_rate", "n_outliers"], ascending=[False, False])

    per_case = (
        out.groupby("case_id")
        .agg(
            n_outlier_blocks=("block", "size"),
            max_z=("z", "max"),
            mean_z=("z", "mean"),
        )
        .reset_index()
        .sort_values(["n_outlier_blocks", "max_z"], ascending=[False, False])
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Outliers", f"{len(out):,}".replace(",", " "))
    k2.metric("Docs touchés", f"{outlier_docs:,}".replace(",", " "))
    k3.metric(
        "Part des annotations atypiques",
        f"{(len(out) / max(1, total_annotations)):.1%}"
    )

    col_out_1, col_out_2 = st.columns([1.3, 1])

    with col_out_1:
        block_for_outliers = st.selectbox(
            "Bloc à inspecter",
            block_display_list,
            index=0,
            key="outlier_block"
        )
        block_code = block_options.loc[block_options["block_display"] == block_for_outliers, "block"].iloc[0]
        block_points = dd[dd["block"] == block_code].copy()
        block_points["is_outlier"] = np.where(block_points["z"] >= zthr, "Outlier", "Normal")

        fig_outliers_map = px.scatter(
            block_points,
            x="cxn",
            y="cyn",
            color="is_outlier",
            size="z",
            hover_data=["case_id", "doc_type", "year", "collection_code", "z"],
            color_discrete_map={"Normal": "#BFC5CC", "Outlier": "#D9534F"},
            title=f"{block_for_outliers} — position des annotations atypiques",
        )
        fig_outliers_map.add_scatter(
            x=[block_points["cx_mean"].iloc[0]],
            y=[block_points["cy_mean"].iloc[0]],
            mode="markers",
            marker=dict(symbol="x", size=16, color="black"),
            name="Centre moyen",
        )
        fig_outliers_map.update_yaxes(autorange="reversed", title="y normalisé")
        fig_outliers_map.update_xaxes(range=[0, 1], title="x normalisé")
        fig_outliers_map.update_layout(legend_title=None, height=560)
        st.plotly_chart(fig_outliers_map, width="stretch")

    with col_out_2:
        fig_block_outliers = px.bar(
            per_block.head(15).sort_values("outlier_rate", ascending=True),
            x="outlier_rate",
            y="block_display",
            orientation="h",
            text="n_outliers",
            color="max_z",
            color_continuous_scale=["#F6D7D4", "#B22222"],
            title="Blocs avec le plus d'anomalies",
        )
        fig_block_outliers.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_block_outliers.update_layout(
            xaxis_title="Part d'annotations atypiques",
            yaxis_title=None,
            coloraxis_colorbar_title="z max",
            height=560,
        )
        fig_block_outliers.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_block_outliers, width="stretch")

    fig_case_outliers = px.scatter(
        per_case.head(80),
        x="n_outlier_blocks",
        y="max_z",
        size="mean_z",
        hover_data=["case_id", "mean_z"],
        title="Documents les plus atypiques",
    )
    fig_case_outliers.update_layout(
        xaxis_title="Nombre de blocs atypiques dans le document",
        yaxis_title="z maximal",
        height=420,
    )
    st.plotly_chart(fig_case_outliers, width="stretch")

    with st.expander("Voir les tables détaillées"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("Top 20 annotations atypiques")
            cols = ["case_id", "doc_type", "year", "collection_code", "is_humatheque", "block_display", "z"]
            cols = [c for c in cols if c in out.columns]
            st.dataframe(out.sort_values("z", ascending=False)[cols].head(20), width="stretch")

        with c2:
            st.write("Docs avec le plus de blocs atypiques")
            st.dataframe(per_case.head(30), width="stretch")

    st.caption("Lecture rapide : plus un point ou une barre est extrême, plus l’écart au comportement habituel du bloc est important.")

# ---- Tab 4: “zones probables”
with tab4:
    st.subheader("Synthèse : zone probable par bloc")
    with st.expander("Comment lire ces graphiques ?"):
        st.markdown(
        """
    Cette vue résume, pour chaque bloc, **sa position moyenne** et **sa zone probable** sur la page.

    **Centroïde moyen**
    → le point représente la position moyenne du centre du bloc.

    **Rectangle**
    → il représente une zone probable autour de cette position moyenne.
    Sa largeur et sa hauteur dépendent de la dispersion observée des centres.

    **Paramètre `k × écart-type`**
    → il contrôle l’ampleur de la zone affichée.
    - valeur faible = zone plus resserrée
    - valeur élevée = zone plus large

    **Graphique de dispersion**
    → il compare les blocs selon leur variabilité horizontale (`sx`) et verticale (`sy`).

    **Interprétation**
    - point centré + petit rectangle = bloc très stable
    - grand rectangle = bloc plus variable
    - valeurs élevées de `sx` ou `sy` = dispersion importante selon l’axe correspondant
        """
        )

    # build a nicer display mapping in the output tables
    label_map = dict(zip(block_options["block"], block_options["block_display"]))
    k = st.slider("Largeur de la zone probable (k × écart-type)", 0.5, 3.0, 1.0, 0.1)

    g = dff.groupby("block").agg(
        cx=("cxn", "mean"),
        cy=("cyn", "mean"),
        sx=("cxn", "std"),
        sy=("cyn", "std"),
        bw_mean=("bw", "mean"),
        bh_mean=("bh", "mean"),
        n=("block", "size")
    ).reset_index()

    g["sx"] = g["sx"].fillna(0)
    g["sy"] = g["sy"].fillna(0)
    g["bw_mean"] = g["bw_mean"].fillna(0)
    g["bh_mean"] = g["bh_mean"].fillna(0)
    g["x0"] = (g["cx"] - k * g["sx"]).clip(0, 1)
    g["x1"] = (g["cx"] + k * g["sx"]).clip(0, 1)
    g["y0"] = (g["cy"] - k * g["sy"]).clip(0, 1)
    g["y1"] = (g["cy"] + k * g["sy"]).clip(0, 1)
    g["zone_w"] = g["x1"] - g["x0"]
    g["zone_h"] = g["y1"] - g["y0"]
    g["block_display"] = g["block"].map(label_map)
    g = g.sort_values("n", ascending=False)

    top_default = min(8, max(1, len(g)))
    selected_blocks = st.multiselect(
        "Blocs affichés sur la carte",
        g["block_display"].tolist(),
        default=g["block_display"].head(top_default).tolist(),
    )
    selected_g = g[g["block_display"].isin(selected_blocks)].copy()

    colA, colB = st.columns([1.4, 1])

    with colA:
        if selected_g.empty:
            st.info("Sélectionnez au moins un bloc pour afficher la carte des zones probables.")
        else:
            fig_probable = px.scatter(
                selected_g,
                x="cx",
                y="cy",
                color="block_display",
                size="n",
                hover_data={
                    "block_display": True,
                    "n": True,
                    "cx": ":.3f",
                    "cy": ":.3f",
                    "sx": ":.3f",
                    "sy": ":.3f",
                    "zone_w": ":.3f",
                    "zone_h": ":.3f",
                },
                title="Centroïdes moyens et zones probables",
            )

            trace_colors = {trace.name: trace.marker.color for trace in fig_probable.data}
            for row in selected_g.itertuples():
                fig_probable.add_shape(
                    type="rect",
                    x0=row.x0,
                    x1=row.x1,
                    y0=row.y0,
                    y1=row.y1,
                    line=dict(color=trace_colors.get(row.block_display, "#444"), width=2),
                    fillcolor="rgba(0,0,0,0)"
                )

            fig_probable.update_yaxes(autorange="reversed", title="y normalisé")
            fig_probable.update_xaxes(range=[0, 1], title="x normalisé")
            fig_probable.update_layout(
                legend_title=None,
                height=650,
            )
            st.plotly_chart(fig_probable, width="stretch")

    with colB:
        fig_dispersion = px.scatter(
            g,
            x="sx",
            y="sy",
            size="n",
            color="block_display",
            hover_data={"cx": ":.3f", "cy": ":.3f", "n": True},
            title="Dispersion des centroïdes par bloc",
        )
        fig_dispersion.update_layout(
            xaxis_title="Dispersion horizontale (sx)",
            yaxis_title="Dispersion verticale (sy)",
            legend_title=None,
            height=650,
        )
        st.plotly_chart(fig_dispersion, width="stretch")

    st.dataframe(
        g[["block_display", "n", "x0", "y0", "x1", "y1", "cx", "cy", "sx", "sy", "bw_mean", "bh_mean"]],
        width="stretch"
    )
    st.caption("Lecture rapide : un petit rectangle autour du centroïde indique une position plutôt régulière ; un grand rectangle indique une position plus variable.")

# ---- Tab 5: “Couverture / données manquantes”    
with tab5:
    st.subheader("Données manquantes par bloc")
    with st.expander("Comment lire ces graphiques ?"):
        st.markdown(
        """
    Cette vue montre **à quelle fréquence chaque bloc est présent ou absent** dans les documents filtrés.

    **Barres empilées**
    → elles comparent la part de documents où le bloc est **présent** et la part où il est **manquant**.

    **Classement par taux de manque**
    → il ordonne les blocs du plus souvent manquant au moins souvent manquant.
    Le texte sur la barre indique le **nombre de documents concernés**.

    **Heatmap bloc × filtre**
    → elle montre comment le taux de données manquantes varie selon une facette choisie (`doc_type`, année, collection, etc.).

    **Bubble chart**
    → la couleur indique le taux de manque, et la taille du point le nombre de documents dans la modalité correspondante.

    **Interprétation**
    - beaucoup de rouge = bloc souvent absent
    - rouge concentré sur certaines modalités = manque dépendant du filtre
    - gros points rouges = problème visible sur un volume important de documents
        """
        )

    total_docs = cases_ff["case_id"].nunique()

    cov = (
        dff.groupby("block")
        .agg(
            n_annotations=("block", "size"),
            n_docs=("case_id", "nunique"),
        )
        .reset_index()
    )
    cov["coverage_rate"] = cov["n_docs"] / max(1, total_docs)
    cov["missing_docs"] = total_docs - cov["n_docs"]
    cov["missing_rate"] = 1 - cov["coverage_rate"]
    cov["block_display"] = cov["block"].map(label_map)
    cov = cov.sort_values(["missing_rate", "missing_docs"], ascending=[False, False])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Docs (filtrés)", f"{total_docs:,}".replace(",", " "))
    c2.metric("Blocs visibles", f"{cov['block'].nunique():,}".replace(",", " "))
    c3.metric(
        "Blocs > 50% manquants",
        f"{(cov['missing_rate'] > 0.5).sum():,}".replace(",", " ")
    )
    c4.metric("Annotations", f"{len(dff):,}".replace(",", " "))

    st.caption(
        "Lecture rapide: plus la part rouge est grande, plus le bloc manque souvent dans le sous-corpus filtré."
    )

    top_n = min(20, max(5, len(cov)))
    top_missing = cov.head(top_n).copy()

    stacked = top_missing[["block_display", "coverage_rate", "missing_rate"]].melt(
        id_vars="block_display",
        value_vars=["coverage_rate", "missing_rate"],
        var_name="status",
        value_name="rate"
    )
    stacked["status"] = stacked["status"].map({
        "coverage_rate": "Présent",
        "missing_rate": "Manquant",
    })

    col_overall_1, col_overall_2 = st.columns([1.3, 1])

    with col_overall_1:
        fig_missing_stack = px.bar(
            stacked,
            x="rate",
            y="block_display",
            color="status",
            orientation="h",
            barmode="stack",
            category_orders={"block_display": top_missing["block_display"].tolist()[::-1]},
            color_discrete_map={"Présent": "#2E8B57", "Manquant": "#D9534F"},
            title=f"Top {top_n} blocs avec le plus de données manquantes",
        )
        fig_missing_stack.update_layout(
            xaxis_title="Part des documents",
            yaxis_title=None,
            legend_title=None,
        )
        fig_missing_stack.update_xaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig_missing_stack, width="stretch")

    with col_overall_2:
        fig_missing_rate = px.bar(
            top_missing.sort_values("missing_rate", ascending=True),
            x="missing_rate",
            y="block_display",
            orientation="h",
            text="missing_docs",
            color="missing_rate",
            color_continuous_scale=["#FDE0DD", "#D9534F"],
            range_color=[0, 1],
            title="Classement par taux de manque",
        )
        fig_missing_rate.update_traces(texttemplate="%{text} docs", textposition="outside")
        fig_missing_rate.update_layout(
            xaxis_title="Taux de données manquantes",
            yaxis_title=None,
            coloraxis_colorbar_title="Manquant",
        )
        fig_missing_rate.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig_missing_rate, width="stretch")

    with st.expander("Voir le détail par bloc"):
        st.dataframe(
            cov[[
                "block_display",
                "n_docs",
                "missing_docs",
                "coverage_rate",
                "missing_rate",
                "n_annotations",
            ]],
            width="stretch"
        )

    st.divider()
    st.subheader("Répartition des données manquantes selon les filtres")

    facet_options = ["doc_type", "memoire_type_code", "is_humatheque", "collection_code", "year"]
    facet = st.selectbox("Filtre analysé", facet_options, index=0)

    if facet in cases_ff.columns and facet in dff.columns:
        facet_cases = cases_ff[["case_id", facet]].copy()
        facet_ann = dff[["case_id", "block", facet]].copy()

        def format_facet_value(value, facet_name: str) -> str:
            if pd.isna(value):
                return "(null)"
            if facet_name == "is_humatheque":
                if value is True:
                    return "Oui"
                if value is False:
                    return "Non"
                return str(value)
            if facet_name == "year":
                try:
                    return str(int(float(value)))
                except (TypeError, ValueError):
                    return str(value)
            return str(value)

        def sort_facet_values(values: list[str], facet_name: str) -> list[str]:
            non_null_values = [v for v in values if v != "(null)"]
            if facet_name == "year":
                sorted_values = sorted(non_null_values, key=lambda v: int(v))
            elif facet_name == "is_humatheque":
                preferred = ["Oui", "Non"]
                sorted_values = [v for v in preferred if v in non_null_values]
                sorted_values += sorted([v for v in non_null_values if v not in preferred])
            else:
                sorted_values = sorted(non_null_values)
            if "(null)" in values:
                sorted_values.append("(null)")
            return sorted_values

        facet_cases["facet_value"] = facet_cases[facet].apply(lambda v: format_facet_value(v, facet))
        facet_ann["facet_value"] = facet_ann[facet].apply(lambda v: format_facet_value(v, facet))

        total_by_f = (
            facet_cases.groupby("facet_value")["case_id"]
            .nunique()
            .reset_index(name="total_docs")
        )

        if total_by_f.empty:
            st.info(f"Aucune valeur exploitable pour '{facet}'.")
        else:
            total_modalities = len(total_by_f)
            show_all_default = facet == "year" and total_modalities <= 100
            show_all_values = st.checkbox(
                "Afficher toutes les modalités",
                value=show_all_default,
                key=f"show_all_{facet}"
            )

            if show_all_values:
                selected_values = total_by_f["facet_value"].tolist()
            else:
                max_values_default = min(
                    total_modalities,
                    12 if facet == "collection_code" else 20
                )
                max_values = st.slider(
                    "Nombre maximal de modalités affichées",
                    min_value=5,
                    max_value=max(5, min(100, total_modalities)),
                    value=max_values_default,
                    step=1,
                    key=f"max_values_{facet}"
                )
                selected_values = (
                    total_by_f
                    .sort_values(["total_docs", "facet_value"], ascending=[False, True])
                    .head(max_values)["facet_value"]
                    .tolist()
                )

            facet_order = sort_facet_values(selected_values, facet)
            total_by_f = total_by_f[total_by_f["facet_value"].isin(facet_order)].copy()
            total_by_f["facet_value"] = pd.Categorical(
                total_by_f["facet_value"],
                categories=facet_order,
                ordered=True
            )
            total_by_f = total_by_f.sort_values("facet_value")

            cov_by = (
                facet_ann.groupby(["facet_value", "block"])["case_id"]
                .nunique()
                .reset_index(name="n_docs")
            )

            full_grid = pd.MultiIndex.from_product(
                [facet_order, cov["block"].tolist()],
                names=["facet_value", "block"]
            ).to_frame(index=False)

            cov_by = full_grid.merge(cov_by, on=["facet_value", "block"], how="left")
            cov_by["n_docs"] = cov_by["n_docs"].fillna(0)
            cov_by = cov_by.merge(total_by_f, on="facet_value", how="left")
            cov_by["coverage_rate"] = cov_by["n_docs"] / cov_by["total_docs"].replace(0, np.nan)
            cov_by["missing_rate"] = 1 - cov_by["coverage_rate"]
            cov_by["missing_docs"] = cov_by["total_docs"] - cov_by["n_docs"]
            cov_by["block_display"] = cov_by["block"].map(label_map)
            cov_by["facet_value"] = pd.Categorical(
                cov_by["facet_value"],
                categories=facet_order,
                ordered=True
            )

            block_order = (
                cov_by.groupby("block_display")["missing_rate"]
                .mean()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

            heatmap_source = cov_by.pivot_table(
                index="block_display",
                columns="facet_value",
                values="missing_rate",
                fill_value=1.0,
                aggfunc="mean"
            )
            heatmap_source = heatmap_source.reindex(block_order)
            heatmap_source = heatmap_source.reindex(columns=facet_order)

            col_facet_1, col_facet_2 = st.columns([1.2, 1])

            with col_facet_1:
                fig_heatmap = px.imshow(
                    heatmap_source,
                    aspect="auto",
                    zmin=0,
                    zmax=1,
                    color_continuous_scale=["#E8F3EC", "#F6C5C0", "#B22222"],
                    title=f"Taux de données manquantes par bloc × {facet}",
                )
                fig_heatmap.update_layout(
                    xaxis_title=facet,
                    yaxis_title="bloc",
                    coloraxis_colorbar_title="Manquant",
                )
                st.plotly_chart(fig_heatmap, width="stretch")

            with col_facet_2:
                bubble = cov_by.copy()
                bubble["block_display"] = pd.Categorical(
                    bubble["block_display"],
                    categories=block_order[::-1],
                    ordered=True
                )
                fig_bubble = px.scatter(
                    bubble,
                    x="facet_value",
                    y="block_display",
                    size="total_docs",
                    color="missing_rate",
                    color_continuous_scale=["#E8F3EC", "#F6C5C0", "#B22222"],
                    range_color=[0, 1],
                    hover_data=["n_docs", "missing_docs", "total_docs"],
                    title="Volume du filtre + taux de manque",
                )
                fig_bubble.update_layout(
                    xaxis_title=facet,
                    yaxis_title=None,
                    coloraxis_colorbar_title="Manquant",
                )
                st.plotly_chart(fig_bubble, width="stretch")

            st.caption(
                "Heatmap: rouge foncé = bloc souvent absent pour cette modalité. "
                "Bubble chart: la taille indique le volume de documents derrière chaque filtre."
            )

            summary = (
                cov_by.groupby("block_display")
                .agg(
                    avg_missing_rate=("missing_rate", "mean"),
                    max_missing_rate=("missing_rate", "max"),
                    min_missing_rate=("missing_rate", "min"),
                    represented_docs=("total_docs", "sum"),
                )
                .reset_index()
            )
            summary["spread"] = summary["max_missing_rate"] - summary["min_missing_rate"]

            st.dataframe(
                summary.sort_values(["avg_missing_rate", "spread"], ascending=False),
                width="stretch"
            )
    else:
        st.info(f"La facette '{facet}' n'est pas disponible dans les données.")

# ---- Tab 6: search
with tab6:
    st.subheader("Recherche ciblée dans les cas et annotations")
    with st.expander("Comment utiliser cette recherche ?"):
        st.markdown(
        """
    Cette vue permet de lancer des **requêtes prêtes à l’emploi** sur les documents filtrés.

    **Bloc**
    → vous choisissez un type de bloc à analyser.

    **Opérateur**
    → il décrit la condition recherchée :
    - **est présent** : le bloc apparaît au moins une fois dans le document
    - **est manquant** : le bloc n’apparaît pas dans le document
    - **est atypique (outlier)** : au moins une annotation du bloc est éloignée de la position habituelle
    - **apparaît plusieurs fois** : le bloc est annoté plusieurs fois dans un même document

    **Résultats**
    → le tableau liste les documents correspondants, et les vignettes permettent un contrôle visuel rapide quand `source_ref` est disponible.
        """
        )

    search_block_display = st.selectbox(
        "Bloc",
        block_display_list,
        index=0,
        key="search_block"
    )
    search_block_code = block_options.loc[
        block_options["block_display"] == search_block_display,
        "block"
    ].iloc[0]

    operator = st.selectbox(
        "Opérateur",
        [
            "est présent",
            "est manquant",
            "est atypique (outlier)",
            "apparaît plusieurs fois",
        ],
        index=0,
        key="search_operator"
    )

    zthr_search = None
    min_count = 2
    if operator == "est atypique (outlier)":
        zthr_search = st.slider(
            "Seuil z pour les outliers",
            1.0,
            8.0,
            3.0,
            0.1,
            key="search_zthr"
        )
    elif operator == "apparaît plusieurs fois":
        min_count = st.number_input(
            "Nombre minimal d'occurrences",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="search_min_count"
        )

    cases_meta = cases_ff.copy()
    block_ann = dff[dff["block"] == search_block_code].copy()

    if operator == "est atypique (outlier)":
        stats_search = block_ann[["cxn", "cyn"]].agg(["mean", "std"])
        cx_mean = stats_search.loc["mean", "cxn"]
        cy_mean = stats_search.loc["mean", "cyn"]
        cx_std = stats_search.loc["std", "cxn"]
        cy_std = stats_search.loc["std", "cyn"]
        cx_std = np.nan if pd.isna(cx_std) or cx_std == 0 else cx_std
        cy_std = np.nan if pd.isna(cy_std) or cy_std == 0 else cy_std
        block_ann["z"] = np.sqrt(
            ((block_ann["cxn"] - cx_mean) / cx_std) ** 2 +
            ((block_ann["cyn"] - cy_mean) / cy_std) ** 2
        )
        block_ann["z"] = block_ann["z"].replace([np.inf, -np.inf], np.nan)
        matched = (
            block_ann[block_ann["z"] >= zthr_search]
            .groupby("case_id")
            .agg(
                n_matches=("layout_annotation_id", "size"),
                max_z=("z", "max"),
                mean_z=("z", "mean"),
            )
            .reset_index()
        )
    elif operator == "apparaît plusieurs fois":
        matched = (
            block_ann.groupby("case_id")
            .agg(n_matches=("layout_annotation_id", "size"))
            .reset_index()
        )
        matched = matched[matched["n_matches"] >= min_count].copy()
    elif operator == "est présent":
        matched = (
            block_ann.groupby("case_id")
            .agg(n_matches=("layout_annotation_id", "size"))
            .reset_index()
        )
    else:
        present_cases = set(block_ann["case_id"].dropna().unique().tolist())
        matched = cases_meta.loc[~cases_meta["case_id"].isin(present_cases), ["case_id"]].copy()
        matched["n_matches"] = 0

    results = cases_meta.merge(matched, on="case_id", how="inner")

    sort_cols = ["case_name", "case_id"]
    ascending = [True, True]
    if "max_z" in results.columns:
        sort_cols = ["max_z", "mean_z", "case_name"]
        ascending = [False, False, True]
    elif "n_matches" in results.columns:
        sort_cols = ["n_matches", "case_name"]
        ascending = [False, True]

    results = results.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Résultats", f"{len(results):,}".replace(",", " "))
    c2.metric("Bloc", search_block_display)
    c3.metric("Opérateur", operator)

    display_cols = [
        "case_id",
        "case_name",
        "doc_type",
        "memoire_type_code",
        "year",
        "collection_code",
        "is_humatheque",
        "n_matches",
        "max_z",
        "mean_z",
        "source_ref",
        "image_url",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    st.dataframe(results[display_cols], width="stretch")

    show_images = st.checkbox("Afficher des vignettes", value=True, key="search_show_images")
    thumb_count = st.slider(
        "Nombre de vignettes",
        min_value=4,
        max_value=50,
        value=8,
        step=4,
        key="search_thumb_count"
    )

    if show_images:
        thumbs = results[results["image_url"].notna()].head(thumb_count).copy()
        if thumbs.empty:
            st.info("Aucune vignette disponible pour ces résultats.")
        else:
            st.subheader("Aperçu visuel")
            n_cols = 4
            for start in range(0, len(thumbs), n_cols):
                row = thumbs.iloc[start:start + n_cols]
                cols = st.columns(n_cols)
                for col, (_, rec) in zip(cols, row.iterrows()):
                    with col:
                        thumb_label = f"{rec['case_id']}\n{rec['source_ref']}" if pd.notna(rec.get("source_ref")) else rec["case_id"]
                        st.image(rec["image_url"], caption=thumb_label, width='stretch')
                        if "n_matches" in rec and pd.notna(rec["n_matches"]):
                            st.caption(f"Occurrences: {int(rec['n_matches'])}")
                        if "max_z" in rec and pd.notna(rec["max_z"]):
                            st.caption(f"z max: {rec['max_z']:.2f}")
