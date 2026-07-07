import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

CAT_COLS = ["mainroad", "guestroom", "basement", "hotwaterheating",
            "airconditioning", "prefarea", "furnishingstatus"]
FEATURES = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom",
            "basement", "hotwaterheating", "parking", "airconditioning",
            "prefarea", "furnishingstatus"]
TARGET = "price"

# Source dataset prices are in Pakistani Rupees (PKR). Converting to Indian
# Rupees (INR) at the mid-market rate (1 PKR ~= 0.34 INR) so displayed prices
# are in INR. Update this constant if you want a different/current rate.
PKR_TO_INR = 0.34


class SimpleLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.mapping_ = {}

    def fit_transform(self, series):
        self.classes_ = np.unique(series)
        self.mapping_ = {val: i for i, val in enumerate(self.classes_)}
        return series.map(self.mapping_)

    def transform(self, val_list):
        return np.array([self.mapping_.get(val, -1) for val in val_list])


class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        X_bias = np.column_stack((np.ones(X_arr.shape[0]), X_arr))
        
        coefs, _, _, _ = np.linalg.lstsq(X_bias, y_arr, rcond=None)
        self.intercept_ = coefs[0]
        self.coef_ = coefs[1:]
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_ + self.intercept_


def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    state = np.random.RandomState(random_state)
    shuffled_indices = state.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
        
    if isinstance(y, pd.Series):
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        y_train, y_test = y[train_indices], y[test_indices]
        
    return X_train, X_test, y_train, y_test


@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    df = df.drop_duplicates()
    for c in CAT_COLS:
        df[c] = df[c].str.lower().str.strip()
    df["price"] = (df["price"] * PKR_TO_INR).round(0).astype(int)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    encoded = df.copy()
    encoders = {}
    for col in CAT_COLS:
        le = SimpleLabelEncoder()
        encoded[col] = le.fit_transform(encoded[col])
        encoders[col] = le

    X = encoded[FEATURES]
    y = encoded[TARGET]
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SimpleLinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    
    # Custom metrics calculations
    y_test_arr = np.asarray(y_test)
    rmse = float(np.sqrt(np.mean((y_test_arr - pred) ** 2)))
    ss_res = np.sum((y_test_arr - pred) ** 2)
    ss_tot = np.sum((y_test_arr - np.mean(y_test_arr)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    mae = float(np.mean(np.abs(y_test_arr - pred)))
    
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
    }
    return model, encoders, metrics


def encode_input(raw: dict, encoders: dict) -> pd.DataFrame:
    row = raw.copy()
    for col in CAT_COLS:
        row[col] = encoders[col].transform([row[col]])[0]
    return pd.DataFrame([row])[FEATURES]


df = load_data()
model, encoders, metrics = train_model(df)

st.title("🏡 House Price Predictor")
st.caption(
    "Linear Regression model trained on the Kaggle Housing Prices dataset "
    f"({len(df)} listings)."
)

tab_predict, tab_insights = st.tabs(["Predict", "Insights"])

# ---------------- PREDICT TAB ----------------
with tab_predict:
    st.subheader("Enter house details")

    area_min, area_max = int(df.area.min()), int(df.area.max())
    bedrooms_min, bedrooms_max = int(df.bedrooms.min()), int(df.bedrooms.max())
    bathrooms_min, bathrooms_max = int(df.bathrooms.min()), int(df.bathrooms.max())
    stories_min, stories_max = int(df.stories.min()), int(df.stories.max())
    parking_min, parking_max = int(df.parking.min()), int(df.parking.max())

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(
            "Area (sq ft)", min_value=area_min, max_value=area_max,
            value=int(df.area.median()), step=50
        )
        bedrooms = st.number_input(
            "Bedrooms", min_value=bedrooms_min, max_value=bedrooms_max,
            value=int(df.bedrooms.median())
        )
        bathrooms = st.number_input(
            "Bathrooms", min_value=bathrooms_min, max_value=bathrooms_max,
            value=int(df.bathrooms.median())
        )
        stories = st.number_input(
            "Stories", min_value=stories_min, max_value=stories_max,
            value=int(df.stories.median())
        )
        parking = st.number_input(
            "Parking spaces", min_value=parking_min, max_value=parking_max,
            value=int(df.parking.median())
        )
    with col2:
        mainroad = st.selectbox("Main road access", ["yes", "no"])
        guestroom = st.selectbox("Guest room", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot water heating", ["yes", "no"])
        airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred area", ["yes", "no"])
        furnishingstatus = st.selectbox(
            "Furnishing status", ["furnished", "semi-furnished", "unfurnished"]
        )

    if st.button("Predict Price", type="primary"):
        raw = dict(
            area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories,
            mainroad=mainroad, guestroom=guestroom, basement=basement,
            hotwaterheating=hotwaterheating, parking=parking,
            airconditioning=airconditioning, prefarea=prefarea,
            furnishingstatus=furnishingstatus,
        )
        X_input = encode_input(raw, encoders)
        pred_price = model.predict(X_input)[0]
        st.success(f"Predicted Price: ₹ {pred_price:,.0f}")
        st.caption(
            f"Model accuracy on held-out test data: R² = {metrics['r2']:.2f}, "
            f"typical error ± ₹{metrics['mae']:,.0f}. Treat this as an estimate, "
            "not an exact valuation."
        )

# ---------------- INSIGHTS TAB ----------------
with tab_insights:
    st.subheader("Dataset & model insights")

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['r2']:.2f}")
    m2.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
    m3.metric("MAE", f"₹{metrics['mae']:,.0f}")

    st.markdown("**Price distribution**")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.histplot(df["price"], bins=30, kde=True, ax=ax1)
    ax1.set_xlabel("Price")
    st.pyplot(fig1)

    st.markdown("**Feature correlation with price**")
    encoded_for_corr = df.copy()
    for col in CAT_COLS:
        encoded_for_corr[col] = encoders[col].transform(encoded_for_corr[col])
    corr = encoded_for_corr.corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    st.markdown("**What drives the prediction (Linear Regression coefficients)**")
    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(
        "Positive coefficients push price up, negative push it down, per one-unit "
        "increase in that feature (categorical features are 0/1 or label-encoded)."
    )
