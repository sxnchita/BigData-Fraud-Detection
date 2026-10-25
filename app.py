import streamlit as st
import tensorflow as tf
import joblib
import json
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    **AI-Based Anomaly Detection System**

    This prototype uses a trained **Autoencoder** to identify
    transactions that differ significantly from learned normal
    transaction patterns.
    """
)


# ============================================================
# LOAD MODEL, SCALER AND THRESHOLD
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/autoencoder.keras"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    with open("models/threshold.json", "r") as f:
        threshold = json.load(f)["threshold"]

    return model, scaler, threshold


@st.cache_data
def load_data():

    return pd.read_csv(
        "data/processed/demo_transactions.csv"
    )


autoencoder, scaler, threshold = load_model()

data = load_data()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.success("✅ Detection system ready")


# ============================================================
# TRANSACTION SELECTION
# ============================================================

st.subheader("🔎 Analyze a Transaction")

st.write(
    "Select a transaction from the dataset to evaluate it "
    "using the trained Autoencoder."
)

transaction_index = st.number_input(
    "Transaction Number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)


# ============================================================
# TRANSACTION INFORMATION
# ============================================================

selected_transaction = data.iloc[int(transaction_index)]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Transaction ID",
        int(transaction_index)
    )

with col2:
    st.metric(
        "Transaction Amount",
        f"${selected_transaction['Amount']:.2f}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    # --------------------------------------------------------
    # Remove target variable
    # --------------------------------------------------------

    features = selected_transaction.drop(
        labels=["Class"]
    )

    # --------------------------------------------------------
    # Convert to DataFrame to preserve feature names
    # --------------------------------------------------------

    features_df = pd.DataFrame(
        [features.values],
        columns=features.index
    )

    # --------------------------------------------------------
    # Scale transaction
    # --------------------------------------------------------

    scaled_features = scaler.transform(
        features_df
    )

    # --------------------------------------------------------
    # Autoencoder reconstruction
    # --------------------------------------------------------

    reconstructed = autoencoder.predict(
        scaled_features,
        verbose=0
    )

    # --------------------------------------------------------
    # Reconstruction error
    # --------------------------------------------------------

    reconstruction_error = float(
        ((scaled_features - reconstructed) ** 2).mean()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    is_anomaly = reconstruction_error > threshold


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("Prediction")

    if is_anomaly:

        st.error(
            "🚨 Potential Anomaly Detected"
        )

        st.write(
            "The transaction has a reconstruction error "
            "above the detection threshold."
        )

    else:

        st.success(
            "✅ Transaction Appears Normal"
        )

        st.write(
            "The transaction falls within the learned "
            "normal reconstruction range."
        )


    # ========================================================
    # SCORES
    # ========================================================

    st.divider()

    score_col, threshold_col = st.columns(2)

    with score_col:

        st.metric(
            "Anomaly Score",
            f"{reconstruction_error:.4f}"
        )

    with threshold_col:

        st.metric(
            "Detection Threshold",
            f"{threshold:.4f}"
        )


    # ========================================================
    # SCORE INTERPRETATION
    # ========================================================

    if is_anomaly:

        st.warning(
            f"Anomaly score ({reconstruction_error:.4f}) "
            f"is greater than the threshold "
            f"({threshold:.4f})."
        )

    else:

        st.info(
            f"Anomaly score ({reconstruction_error:.4f}) "
            f"is below the threshold "
            f"({threshold:.4f})."
        )


# ============================================================
# ABOUT THE MODEL
# ============================================================

with st.expander("ℹ️ About the Detection System"):

    st.write(
        """
        The system uses an Autoencoder-based anomaly detection
        approach.

        The Autoencoder learns patterns from normal transactions
        and attempts to reconstruct incoming transactions.

        A reconstruction error is calculated between the original
        and reconstructed transaction.

        If the reconstruction error exceeds the selected threshold,
        the transaction is flagged as a potential anomaly.
        """
    )

    st.write(
        f"**Detection threshold:** {threshold:.4f}"
    )

    st.write(
        "**Model input features:** 30"
    )