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

st.set_page_config(page_title="VLM Layout Explorer", layout="wide")

# -----------------------
# Env / DB config
# -----------------------
load_dotenv()  # loads .env in current working dir
DB_URL = os.getenv("DATABASE_URL", "").strip()

if not DB_URL:
    st.error("DB_URL not found. Add it to your .env file, e.g. DB_URL=postgresql+psycopg2://user:pass@host:5432/db")
    st.stop()

# Optional: set schema name here if not "vlm_eval"
DB_SCHEMA = os.getenv("DB_SCHEMA", "vlm_eval").strip()

# -----------------------
# Data loader
# -----------------------
@st.cache_data(show_spinner=False)
def load_from_db(db_url: str, schema: str) -> pd.DataFrame:
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

    q = text(f"""
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
    with engine.begin() as cx:
        df = pd.read_sql(q, cx)

    # typing
    for col in ["x1n", "y1n", "x2n", "y2n", "cxn", "cyn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "is_humatheque" in df.columns:
        # keep boolean, but tolerate strings
        if df["is_humatheque"].dtype == object:
            df["is_humatheque"] = df["is_humatheque"].astype(str).str.lower().map({"true": True, "false": False})
    return df

def ensure_bbox(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bw"] = (df["x2n"] - df["x1n"]).clip(lower=0)
    df["bh"] = (df["y2n"] - df["y1n"]).clip(lower=0)
    df["barea"] = (df["bw"] * df["bh"]).clip(lower=0)
    return df

df = load_from_db(DB_URL, DB_SCHEMA)
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

# doc_type facet
doc_types = sorted([x for x in df["doc_type"].dropna().unique().tolist()])
if doc_types:
    sel_doc_types = st.sidebar.multiselect("doc_type", doc_types, default=doc_types)
    mask &= df["doc_type"].isin(sel_doc_types)

# memoire_type_code facet
memoire_types = sorted([x for x in df["memoire_type_code"].dropna().unique().tolist()])
if df["memoire_type_code"].isna().any():
    memoire_types = memoire_types + ["(null)"]
if memoire_types:
    sel_memoire_types = st.sidebar.multiselect("memoire_type_code", memoire_types, default=memoire_types)
    memoire_mask = df["memoire_type_code"].isin([x for x in sel_memoire_types if x != "(null)"])
    if "(null)" in sel_memoire_types:
        memoire_mask |= df["memoire_type_code"].isna()
    mask &= memoire_mask

# is_humatheque facet
if df["is_humatheque"].notna().any():
    sel_huma = st.sidebar.selectbox("is_humatheque", ["(tous)", True, False], index=0)
    if sel_huma != "(tous)":
        mask &= (df["is_humatheque"] == sel_huma)

# collection_code facet
collections = sorted([x for x in df["collection_code"].dropna().unique().tolist()])
if collections:
    sel_cols = st.sidebar.multiselect("collection_code", collections, default=collections)
    mask &= df["collection_code"].isin(sel_cols)

# year facet (range)
if df["year"].notna().any():
    y_min = int(np.nanmin(df["year"]))
    y_max = int(np.nanmax(df["year"]))
    yr = st.sidebar.slider("Années", min_value=y_min, max_value=y_max, value=(y_min, y_max))
    mask &= df["year"].between(yr[0], yr[1])

dff = df.loc[mask].copy()

# -----------------------
# Header KPIs
# -----------------------
st.title("VLM Layout Explorer — stats sur blocs annotés (DB)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Annotations (lignes)", f"{len(dff):,}".replace(",", " "))
c2.metric("Docs (case_id)", f"{dff['case_id'].nunique():,}".replace(",", " "))
c3.metric("Types de blocs", f"{dff['block'].nunique():,}".replace(",", " "))
c4.metric("Campagnes", f"{dff['campaign_id'].nunique():,}".replace(",", " "))

st.divider()

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Carte des positions",
    "Distributions",
    "Qualité / outliers",
    "Synthèse 'zones probables'",
    "Couverture / données manquantes"
])

# ---- Tab 1: positions map
with tab1:
    st.subheader("Nuage de centroïdes (0..1) + densité")

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

    Dans notre contexte cela peut indiquer :

    - pages de titre atypiques  
    - documents mal structurés   
    - cas difficiles pour l’extraction automatique

    ### Interprétation pour le projet VLM

    Un bloc est **facile à extraire automatiquement** lorsque :

    - la **boîte est étroite** (position stable)
    - il y a **peu d’outliers**
    - la médiane est clairement identifiable

    À l’inverse, une distribution **large ou très dispersée** indique que :

    → la position du bloc varie fortement selon les documents.

    Dans ce cas, l'extraction devra être :

    - soit plus robuste (VLM global)
    - soit guidée par des heuristiques supplémentaires.
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

    st.caption("Lecture : un bloc 'stable' = distribution serrée ⇒ bon candidat pour extraction guidée / cropping.")

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

    stats = dff.groupby("block")[["cxn", "cyn"]].agg(["mean", "std"]).reset_index()
    stats.columns = ["block", "cx_mean", "cx_std", "cy_mean", "cy_std"]

    dd = dff.merge(stats, on="block", how="left")
    dd["cx_std"] = dd["cx_std"].replace(0, np.nan)
    dd["cy_std"] = dd["cy_std"].replace(0, np.nan)
    dd["z"] = np.sqrt(((dd["cxn"] - dd["cx_mean"]) / dd["cx_std"])**2 + ((dd["cyn"] - dd["cy_mean"]) / dd["cy_std"])**2)
    dd["z"] = dd["z"].replace([np.inf, -np.inf], np.nan)

    zthr = st.slider("Seuil z (distance normalisée)", 1.0, 8.0, 3.0, 0.1)
    out = dd[dd["z"] >= zthr].copy()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Outliers", f"{len(out):,}".replace(",", " "))
        st.write("Top 20 (z décroissant)")
        cols = ["case_id", "doc_type", "year", "collection_code", "is_humatheque", "block_display", "z"]
        cols = [c for c in cols if c in out.columns]
        st.dataframe(out.sort_values("z", ascending=False)[cols].head(20), width="stretch")

    with c2:
        per_case = out.groupby("case_id").size().sort_values(ascending=False).reset_index(name="n_outlier_blocks")
        st.write("Docs avec le plus de blocs atypiques (souvent : scans bizarres, page non-standard, etc.)")
        st.dataframe(per_case.head(30), width="stretch")

    st.caption("Usage VLM : les outliers = cas de robustesse. Tu peux décider d'un fallback (prompt global, OCR, ou extraction par zones).")

# ---- Tab 4: “zones probables”
with tab4:
    st.subheader("Synthèse : zone probable par bloc (pour guidage/cropping)")

    method = st.radio("Méthode", ["Quantiles (robuste)", "Moyenne ± écart-type"], horizontal=True)

    # build a nicer display mapping in the output tables
    label_map = dict(zip(block_options["block"], block_options["block_display"]))

    if method.startswith("Quantiles"):
        q = st.slider("Quantile (enveloppe)", 0.50, 0.99, 0.90, 0.01)

        g = dff.groupby("block").agg(
            # zone probable via quantiles bbox
            x1n_low=("x1n", lambda s: np.nanquantile(s, 1 - q)),
            y1n_low=("y1n", lambda s: np.nanquantile(s, 1 - q)),
            x2n_high=("x2n", lambda s: np.nanquantile(s, q)),
            y2n_high=("y2n", lambda s: np.nanquantile(s, q)),

            # centroïdes probables
            cx_q50=("cxn", lambda s: np.nanquantile(s, 0.50)),
            cy_q50=("cyn", lambda s: np.nanquantile(s, 0.50)),
            cx_mean=("cxn", "mean"),
            cy_mean=("cyn", "mean"),
            cx_std=("cxn", "std"),
            cy_std=("cyn", "std"),

            n=("block", "size"),
        ).reset_index()

        g["block_display"] = g["block"].map(label_map)
        g = g.sort_values("n", ascending=False)
        st.dataframe(
            g[[
                "block_display", "n",
                "x1n_low", "y1n_low", "x2n_high", "y2n_high",
                "cx_q50", "cy_q50", "cx_mean", "cy_mean", "cx_std", "cy_std"
            ]],
            width="stretch"
        )

    else:
        k = st.slider("k (écart-type)", 0.5, 3.0, 1.0, 0.1)

        g = dff.groupby("block").agg(
            cx=("cxn", "mean"),
            cy=("cyn", "mean"),
            sx=("cxn", "std"),
            sy=("cyn", "std"),
            n=("block", "size")
        ).reset_index()

        g["x0"] = (g["cx"] - k * g["sx"]).clip(0, 1)
        g["x1"] = (g["cx"] + k * g["sx"]).clip(0, 1)
        g["y0"] = (g["cy"] - k * g["sy"]).clip(0, 1)
        g["y1"] = (g["cy"] + k * g["sy"]).clip(0, 1)

        g["block_display"] = g["block"].map(label_map)
        g = g.sort_values("n", ascending=False)

        st.dataframe(
            g[["block_display", "n", "x0", "y0", "x1", "y1", "cx", "cy", "sx", "sy"]],
            width="stretch"
        )

    st.caption("Ces zones probables sont directement réutilisables pour un pipeline VLM 'cropped' (crop → extraction champ par champ).")

# ---- Tab 5: “Couverture / données manquantes”    
with tab5:
    st.subheader("Couverture des blocs (présence/absence)")

    # nombre total de docs dans le sous-corpus filtré
    total_docs = dff["case_id"].nunique()

    # coverage par bloc : nb docs où le bloc est présent
    cov = (
        dff.groupby("block")
           .agg(
               n_annotations=("block", "size"),
               n_docs=("case_id", "nunique"),
               cx_mean=("cxn", "mean"),
               cy_mean=("cyn", "mean"),
           )
           .reset_index()
    )
    cov["coverage_rate"] = cov["n_docs"] / max(1, total_docs)
    cov["block_display"] = cov["block"].map(label_map)
    cov = cov.sort_values(["coverage_rate", "n_docs"], ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Docs (filtrés)", f"{total_docs:,}".replace(",", " "))
    c2.metric("Blocs uniques", f"{cov['block'].nunique():,}".replace(",", " "))
    c3.metric("Annotations", f"{len(dff):,}".replace(",", " "))

    st.divider()

    # Chart 1: coverage rate
    fig1 = px.bar(
        cov,
        x="block_display",
        y="coverage_rate",
        title="Taux de couverture par bloc (docs où le bloc apparaît / total docs)",
    )
    fig1.update_layout(xaxis_title=None, yaxis_title="coverage_rate (0..1)")
    st.plotly_chart(fig1, width="stretch")

    # Chart 2: n_docs
    fig2 = px.bar(
        cov,
        x="block_display",
        y="n_docs",
        title="Nombre de documents où le bloc est présent",
    )
    fig2.update_layout(xaxis_title=None, yaxis_title="n_docs")
    st.plotly_chart(fig2, width="stretch")

    docs_meta = dff[["case_id", "memoire_type_code"]].drop_duplicates()
    docs_meta["memoire_type_code_display"] = docs_meta["memoire_type_code"].fillna("(null)")
    if docs_meta["memoire_type_code"].notna().any():
        memoire_counts = (
            docs_meta.groupby("memoire_type_code_display")["case_id"]
            .nunique()
            .reset_index(name="n_docs")
            .sort_values(["n_docs", "memoire_type_code_display"], ascending=[False, True])
        )
        fig_memoire = px.bar(
            memoire_counts,
            x="memoire_type_code_display",
            y="n_docs",
            title="Documents filtrÃ©s par memoire_type_code",
        )
        fig_memoire.update_layout(xaxis_title="memoire_type_code", yaxis_title="n_docs")
        st.plotly_chart(fig_memoire, width="stretch")

    st.divider()
    st.subheader("Couverture par facette")

    facet_options = ["doc_type", "is_humatheque", "collection_code", "year"]
    if dff["memoire_type_code"].notna().any():
        facet_options.append("memoire_type_code")

    facet = st.selectbox("Facette", facet_options, index=0)

    if facet in dff.columns:
        # coverage rate par bloc *et* facette
        # calc : docs total par facette
        total_by_f = dff.groupby(facet)["case_id"].nunique().reset_index(name="total_docs")

        # docs couverts par bloc et facette
        cov_by = (
            dff.groupby([facet, "block"])["case_id"].nunique()
               .reset_index(name="n_docs")
        )
        cov_by = cov_by.merge(total_by_f, on=facet, how="left")
        cov_by["coverage_rate"] = cov_by["n_docs"] / cov_by["total_docs"].replace(0, np.nan)
        cov_by["block_display"] = cov_by["block"].map(label_map)

        # pour éviter un graphe illisible sur collection_code/year
        if facet in ["collection_code", "year"]:
            top = total_by_f.sort_values("total_docs", ascending=False).head(20)[facet].tolist()
            cov_by = cov_by[cov_by[facet].isin(top)]

        # un bloc choisi pour lecture claire
        one_block_display = st.selectbox("Bloc", block_display_list, index=0, key="cov_block")
        one_block_code = block_options.loc[block_options["block_display"] == one_block_display, "block"].iloc[0]
        dd = cov_by[cov_by["block"] == one_block_code].copy()

        fig3 = px.bar(
            dd,
            x=facet,
            y="coverage_rate",
            title=f"Couverture de '{one_block_display}' selon {facet}",
        )
        fig3.update_layout(xaxis_title=None, yaxis_title="coverage_rate (0..1)")
        st.plotly_chart(fig3, width="stretch")

        # table aussi
        st.dataframe(
            dd.sort_values("coverage_rate", ascending=False)[[facet, "n_docs", "total_docs", "coverage_rate"]],
            width="stretch"
        )
    else:
        st.info(f"La facette '{facet}' n'est pas disponible dans les données.")
        
    st.divider()
    st.subheader("Matrice de couverture (heatmap) — blocs × facette")

    facet2 = st.selectbox(
        "Facette (heatmap)",
        ["year", "doc_type", "is_humatheque", "collection_code"],
        index=0,
        key="heatmap_facet"
    )

    if facet2 in dff.columns:

        # total docs par facette
        total_by_f = (
            dff.groupby(facet2)["case_id"]
            .nunique()
            .reset_index(name="total_docs")
        )

        # coverage docs par bloc et facette
        cov_by = (
            dff.groupby([facet2, "block"])["case_id"]
            .nunique()
            .reset_index(name="n_docs")
        )

        cov_by = cov_by.merge(total_by_f, on=facet2, how="left")

        cov_by["coverage_rate"] = (
            cov_by["n_docs"] /
            cov_by["total_docs"].replace(0, np.nan)
        )

        cov_by["block_display"] = cov_by["block"].map(label_map)

        # pivot → matrice
        mat = cov_by.pivot_table(
            index="block_display",
            columns=facet2,
            values="coverage_rate",
            fill_value=0.0,
            aggfunc="mean"
        )

        # tri blocs par couverture moyenne
        mat["__mean__"] = mat.mean(axis=1)
        mat = mat.sort_values("__mean__", ascending=False)
        mat = mat.drop(columns="__mean__")

        # tri colonnes par nombre de docs
        col_order = (
            total_by_f
            .sort_values("total_docs", ascending=False)[facet2]
            .tolist()
        )

        mat = mat[col_order]

        fig_hm = px.imshow(
            mat,
            aspect="auto",
            title=f"Couverture (taux) — blocs × {facet2}",
            zmin=0,
            zmax=1
        )

        fig_hm.update_layout(
            xaxis_title=facet2,
            yaxis_title="bloc",
        )

        st.plotly_chart(fig_hm, width="stretch")

        with st.expander("Voir la table (coverage_rate pivot)"):
            st.dataframe(mat.reset_index(), width="stretch")

        st.caption(
            "0 = bloc absent dans ces documents ; 1 = bloc quasi systématique. "
            "Permet d'identifier les blocs rares ou dépendants d'une facette."
        )

    else:
        st.info(f"La facette '{facet2}' n'est pas disponible dans les données.")
