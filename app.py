import streamlit as st
import pandas as pd
import joblib

# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load(r"D:\Project DS Final\Classi_Models\xgboost.pkl")

model = load_model()

# === App Title ===
st.title("✈️ Flight Delay Prediction Demo")
st.write("Nhập thông tin chuyến bay để dự đoán khả năng bị delay (>15 phút).")

# === Input Form ===
col1, col2 = st.columns(2)

with col1:
    origin = st.selectbox("Sân bay đi (ORIGIN):", ["SEA", "GEG", "PAE", "PSC"])
    dep_time = st.number_input("Giờ khởi hành (HH.MM):", 0.0, 24.0, 9.30)
    distance = st.number_input("Khoảng cách (miles):", 0, 3000, 800)

with col2:
    carrier = st.selectbox("Hãng bay:", ["AA", "DL", "UA", "AS"])
    arr_time = st.number_input("Giờ đến dự kiến (HH.MM):", 0.0, 24.0, 12.15)
    temperature = st.number_input("Nhiệt độ (°C):", -20.0, 50.0, 20.0)

# === Khi người dùng nhấn nút dự đoán ===
if st.button("🔍 Dự đoán"):
    # Chuẩn bị dữ liệu đầu vào
    input_data = pd.DataFrame({
        "OP_UNIQUE_CARRIER": [carrier],
        "ORIGIN": [origin],
        "DEP_TIME": [dep_time],
        "ARR_TIME": [arr_time],
        "DISTANCE": [distance],
        "TEMP": [temperature]
    })

    # (nếu có preprocess)
    # from utils.preprocess import transform_input
    # input_data = transform_input(input_data)

    # Dự đoán
    prediction = model.predict(input_data)[0]
    result = "🚫 Delay (>15 phút)" if prediction == 1 else "✅ Không delay"

    # Hiển thị kết quả
    st.subheader("Kết quả dự đoán:")
    st.success(result)
