
# Run with: streamlit run app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "House Price Predictor",
    page_icon  = "🏠",
    layout     = "wide"
)

API_URL = "http://localhost:8000"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("Predict Ames, Iowa residential house prices using CatBoost.")

# ── Sidebar: model info ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    try:
        info = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("✅ API Online")

        model_info = info.get('model', {})
        st.markdown(f"**Model:** {model_info.get('model_class', 'N/A')}")
        st.markdown(f"**RMSE:** ${model_info.get('RMSE', 0):,.0f}")
        st.markdown(f"**MAE:** ${model_info.get('MAE', 0):,.0f}")
        st.markdown(f"**MAPE:** {model_info.get('MAPE', 0):.2f}%")
        st.markdown(f"**R²:** {model_info.get('R2', 0):.4f}")
        st.markdown("---")
        st.markdown("*Data: Ames, Iowa 2006–2010*")
        st.markdown("*Target: SalePrice (USD)*")

    except Exception:
        st.error("❌ API Offline")
        st.code("uvicorn api.main:app --port 8000")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction",
                              "📦 Batch Prediction",
                              "📖 API Info"])


# ═════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ═════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict price for a single house")
    st.markdown("Adjust the sliders and dropdowns, then click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Size & Layout**")
        GrLivArea   = st.slider("Above-ground living area (sqft)",
                                  500, 5000, 1500, step=50)
        TotalBsmtSF = st.slider("Total basement area (sqft)",
                                  0, 3000, 800, step=50)
        GarageArea  = st.slider("Garage area (sqft)",
                                  0, 1500, 400, step=50)
        Total_Porch = st.slider("Total porch area (sqft)",
                                  0, 600, 50, step=10)

    with col2:
        st.markdown("**Quality & Condition**")
        OverallQual = st.selectbox("Overall quality (1-10)",
                                    options=list(range(1, 11)),
                                    index=6)
        OverallCond = st.selectbox("Overall condition (1-10)",
                                    options=list(range(1, 11)),
                                    index=4)
        KitchenQual = st.selectbox("Kitchen quality",
                                    options=[1, 2, 3, 4, 5],
                                    format_func=lambda x: {
                                        1: 'Poor', 2: 'Fair',
                                        3: 'Typical', 4: 'Good',
                                        5: 'Excellent'}[x],
                                    index=2)
        ExterQual   = st.selectbox("Exterior quality",
                                    options=[1, 2, 3, 4, 5],
                                    format_func=lambda x: {
                                        1: 'Poor', 2: 'Fair',
                                        3: 'Typical', 4: 'Good',
                                        5: 'Excellent'}[x],
                                    index=2)

    with col3:
        st.markdown("**Age & Features**")
        YearBuilt   = st.slider("Year built",
                                  1880, 2010, 1990)
        YearRemod   = st.slider("Year remodeled",
                                  1950, 2010, 1990)
        GarageCars  = st.selectbox("Garage capacity (cars)",
                                    options=[0, 1, 2, 3, 4],
                                    index=2)
        FullBath    = st.selectbox("Full bathrooms",
                                    options=[0, 1, 2, 3, 4],
                                    index=2)
        Fireplaces  = st.selectbox("Fireplaces",
                                    options=[0, 1, 2, 3],
                                    index=0)

    # ── Derived features ──────────────────────────────────────────────────────
    # These mirror what notebook 02 created
    TotalSF          = TotalBsmtSF + GrLivArea
    Quality_x_Area   = OverallQual * GrLivArea
    Quality_x_TotalSF = OverallQual * TotalSF
    HouseAge         = 2010 - YearBuilt
    RemodAge         = 2010 - YearRemod
    IsRemodeled      = int(YearBuilt != YearRemod)
    HasGarage        = int(GarageArea > 0)
    HasFireplace     = int(Fireplaces > 0)
    HasPorch         = int(Total_Porch > 0)

    input_data = {
        "GrLivArea":          GrLivArea,
        "TotalBsmtSF":        TotalBsmtSF,
        "GarageArea":         GarageArea,
        "GarageCars":         GarageCars,
        "OverallQual":        OverallQual,
        "OverallCond":        OverallCond,
        "KitchenQual":        KitchenQual,
        "ExterQual":          ExterQual,
        "YearBuilt":          YearBuilt,
        "YearRemodAdd":       YearRemod,
        "FullBath":           FullBath,
        "Fireplaces":         Fireplaces,
        "TotalSF":            TotalSF,
        "Quality_x_Area":     Quality_x_Area,
        "Quality_x_TotalSF":  Quality_x_TotalSF,
        "HouseAge":           HouseAge,
        "RemodAge":           RemodAge,
        "IsRemodeled":        IsRemodeled,
        "HasGarage":          HasGarage,
        "HasFireplace":       HasFireplace,
        "HasPorch":           HasPorch,
        "Total_Porch_SF":     Total_Porch,
    }

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("🔮 Predict Price", type="primary",
                  use_container_width=True):
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"data": input_data},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    price  = result['predicted_price']

                    st.success("✅ Prediction complete")

                    # ── Big price display ─────────────────────────────────────
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px;
                                background: #f0f8f0; border-radius: 10px;
                                border: 2px solid #2ecc71;'>
                        <h1 style='color: #27ae60; font-size: 3em;'>
                            {result['predicted_price_formatted']}
                        </h1>
                        <p style='color: #666;'>Predicted Sale Price</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")

                    # ── Confidence range ──────────────────────────────────────
                    # MAPE is 8.33% so show ±8.33% range
                    mape_pct = 0.0833
                    low      = price * (1 - mape_pct)
                    high     = price * (1 + mape_pct)

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Lower bound (~8% below)",
                                  f"${low:,.0f}")
                    col_b.metric("Predicted price",
                                  result['predicted_price_formatted'])
                    col_c.metric("Upper bound (~8% above)",
                                  f"${high:,.0f}")

                    st.caption(f"Model: {result['model']}  |  "
                                f"Latency: {result['latency_ms']:.1f}ms  |  "
                                f"Typical error: ±8.33%")

                    with st.expander("Show input features sent to model"):
                        st.json(input_data)

                    with st.expander("Show raw API response"):
                        st.json(result)

                else:
                    st.error(f"API error {response.status_code}: "
                              f"{response.text}")

            except requests.ConnectionError:
                st.error("Cannot reach the API.")
                st.code("uvicorn api.main:app --port 8000",
                         language="bash")


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Predict prices for multiple houses")
    st.markdown("Upload a CSV file. Each row is one house.")

    st.info("The CSV should contain the feature columns produced by "
             "notebook 02. Any missing columns are filled with 0.")

    uploaded = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df):,}** houses, "
                  f"**{len(df.columns)}** features")
        st.dataframe(df.head())

        if st.button("🔮 Predict All Houses", type="primary"):
            with st.spinner(f"Predicting {len(df):,} houses..."):
                try:
                    records  = df.to_dict(orient='records')
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"data": records},
                        timeout=120
                    )
                    if response.status_code == 200:
                        result = response.json()
                        df['predicted_price']           = result['predictions']
                        df['predicted_price_formatted'] = result['formatted']

                        st.success(
                            f"✅ {result['n_houses']:,} predictions in "
                            f"{result['latency_ms']:.0f}ms")

                        # Summary stats
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Median price",
                                     f"${df['predicted_price'].median():,.0f}")
                        col2.metric("Mean price",
                                     f"${df['predicted_price'].mean():,.0f}")
                        col3.metric("Min price",
                                     f"${df['predicted_price'].min():,.0f}")
                        col4.metric("Max price",
                                     f"${df['predicted_price'].max():,.0f}")

                        st.dataframe(df)

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download predictions CSV",
                            data     = csv,
                            file_name = "house_price_predictions.csv",
                            mime     = "text/csv",
                            use_container_width = True
                        )
                    else:
                        st.error(f"API error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        # Quick demo with sample data
        st.markdown("---")
        st.markdown("**Or try a quick demo with sample data:**")
        if st.button("Run demo with 5 sample houses"):
            sample = [
                {"OverallQual": 9, "GrLivArea": 2800, "GarageCars": 3,
                 "TotalBsmtSF": 1400, "FullBath": 3, "YearBuilt": 2005},
                {"OverallQual": 7, "GrLivArea": 1600, "GarageCars": 2,
                 "TotalBsmtSF": 900,  "FullBath": 2, "YearBuilt": 1998},
                {"OverallQual": 5, "GrLivArea": 1100, "GarageCars": 1,
                 "TotalBsmtSF": 600,  "FullBath": 1, "YearBuilt": 1975},
                {"OverallQual": 8, "GrLivArea": 2200, "GarageCars": 2,
                 "TotalBsmtSF": 1100, "FullBath": 2, "YearBuilt": 2001},
                {"OverallQual": 4, "GrLivArea": 900,  "GarageCars": 1,
                 "TotalBsmtSF": 400,  "FullBath": 1, "YearBuilt": 1955},
            ]
            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    json={"data": sample},
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    demo_df = pd.DataFrame(sample)
                    demo_df['predicted_price'] = result['formatted']
                    st.dataframe(demo_df)
            except Exception as e:
                st.error(f"Error: {e}")


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — API Info
# ═════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("API Documentation")
    st.markdown(f"""
    The REST API runs at `{API_URL}`

    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | GET | `/` | Welcome message |
    | GET | `/health` | Health check + model metrics |
    | GET | `/info` | Model metadata |
    | POST | `/predict` | Single house prediction |
    | POST | `/predict/batch` | Multi-house prediction |

    **Interactive docs (Swagger UI):** [{API_URL}/docs]({API_URL}/docs)

    ---
    **Example — single prediction:**
```bash
    curl -X POST {API_URL}/predict \\
         -H "Content-Type: application/json" \\
         -d '{{"data": {{"OverallQual": 8, "GrLivArea": 2000}}}}'
```

    **Example — batch prediction:**
```bash
    curl -X POST {API_URL}/predict/batch \\
         -H "Content-Type: application/json" \\
         -d '{{"data": [{{"OverallQual": 7, "GrLivArea": 1500}},
                        {{"OverallQual": 9, "GrLivArea": 3000}}]}}'
```
    """)
