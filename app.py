import streamlit as st
import pandas as pd
import joblib
import datetime as dt

model = joblib.load(r"D:\Project DS Final\Classi_Models\xgboost.pkl")

st.set_page_config(page_title="Flight Delay Demo", layout="centered")
st.title("✈️ Flight Delay Prediction App")
st.caption("Demo dự đoán khả năng chuyến bay bị delay > 15 phút")

# ==========================================================
# 2️⃣ Nhập thông tin chuyến bay
# ==========================================================
st.subheader("🛫 Nhập thông tin chuyến bay")

col1, col2 = st.columns(2)
with col1:
    op_unique_carrier = st.selectbox("Hãng bay (Carrier)", 
                                     ["AA","AS","B6","DL","F9","G4","HA","MQ","NK","OO","UA","WN"])
    origin = st.selectbox("Sân bay đi (Origin)", 
                          ["BLI","GEG","PAE","PSC","SEA"])
with col2:
    dest = st.selectbox("Sân bay đến (Destination)", 
                        ["ABQ","ANC","ATL","AUS","AZA","BLI","BNA","BOI","BOS","BUR","BWI","BZN",
                         "CHS","CLE","CLT","CMH","CVG","DAL","DCA","DEN","DFW","DTW","EUG","EWR",
                         "FAI","FAT","FLL","GEG","GTF","HLN","HNL","HOU","IAD","IAH","IND","JAC",
                         "JFK","JNU","KOA","KTN","LAS","LAX","LIH","LWS","MCI","MCO","MDW","MFR",
                         "MIA","MKE","MRY","MSO","MSP","MSY","OAK","OGG","OKC","OMA","ONT","ORD",
                         "PDX","PHL","PHX","PIT","PSC","PSP","RDD","RDM","RDU","RNO","RSW","SAN",
                         "SAT","SBA","SBP","SEA","SFO","SIT","SJC","SLC","SMF","SNA","STL","STS",
                         "TPA","TUS"])
    flight_date = st.date_input("Ngày bay", dt.date.today())

dep_hour = st.slider("Giờ khởi hành (0–23)", 0, 23, 10)

# ==========================================================
# 3️⃣ Mapping thủ công (giống LabelEncoder cũ)
# ==========================================================
carrier_map = {
    "AA": 0, "AS": 1, "B6": 2, "DL": 3, "F9": 4, "G4": 5, "HA": 6,
    "MQ": 7, "NK": 8, "OO": 9, "UA": 10, "WN": 11
}

origin_map = {
    "BLI": 0, "GEG": 1, "PAE": 2, "PSC": 3, "SEA": 4
}

dest_map = {
    "ABQ": 0, "ANC": 1, "ATL": 2, "AUS": 3, "AZA": 4, "BLI": 5, "BNA": 6, "BOI": 7, "BOS": 8, "BUR": 9, "BWI": 10,
    "BZN": 11, "CHS": 12, "CLE": 13, "CLT": 14, "CMH": 15, "CVG": 16, "DAL": 17, "DCA": 18, "DEN": 19, "DFW": 20,
    "DTW": 21, "EUG": 22, "EWR": 23, "FAI": 24, "FAT": 25, "FLL": 26, "GEG": 27, "GTF": 28, "HLN": 29, "HNL": 30,
    "HOU": 31, "IAD": 32, "IAH": 33, "IND": 34, "JAC": 35, "JFK": 36, "JNU": 37, "KOA": 38, "KTN": 39, "LAS": 40,
    "LAX": 41, "LIH": 42, "LWS": 43, "MCI": 44, "MCO": 45, "MDW": 46, "MFR": 47, "MIA": 48, "MKE": 49, "MRY": 50,
    "MSO": 51, "MSP": 52, "MSY": 53, "OAK": 54, "OGG": 55, "OKC": 56, "OMA": 57, "ONT": 58, "ORD": 59, "PDX": 60,
    "PHL": 61, "PHX": 62, "PIT": 63, "PSC": 64, "PSP": 65, "RDD": 66, "RDM": 67, "RDU": 68, "RNO": 69, "RSW": 70,
    "SAN": 71, "SAT": 72, "SBA": 73, "SBP": 74, "SEA": 75, "SFO": 76, "SIT": 77, "SJC": 78, "SLC": 79, "SMF": 80,
    "SNA": 81, "STL": 82, "STS": 83, "TPA": 84, "TUS": 85
}

# ==========================================================
# 4️⃣ Chuẩn bị dữ liệu đầu vào cho mô hình
# ==========================================================
input_data = pd.DataFrame({
    "MONTH": [flight_date.month],
    "DAY_OF_MONTH": [flight_date.day],
    "DAY_OF_WEEK": [flight_date.weekday() + 1],
    "CRS_DEP_TIME": [dep_hour * 100],      # kiểu HHMM
    "CRS_ELAPSED_TIME": [90.0],
    "OP_UNIQUE_CARRIER": [carrier_map[op_unique_carrier]],
    "ORIGIN": [origin_map[origin]],
    "DEST": [dest_map[dest]],
    "DISTANCE": [500.0],
    "HourlyDewPointTemperature": [10.0],
    "HourlyDryBulbTemperature": [20.0],
    "HourlyRelativeHumidity": [60.0],
    "HourlyVisibility": [8.0],
    "HourlyWindSpeed": [5.0],
})

# ==========================================================
# 5️⃣ Dự đoán
# ==========================================================
if st.button("🚀 Dự đoán"):
    # kiểm tra hợp lệ
    if -1 in input_data.values:
        st.error("⚠️ Một hoặc nhiều giá trị không có trong mapping (hãng bay hoặc sân bay không hợp lệ).")
        st.stop()

    y_pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Kết quả dự đoán")
    if y_pred == 1:
        st.error(f"✈️ Chuyến bay có khả năng **Delay**.\nXác suất delay: **{prob:.2%}**")
    else:
        st.success(f"🕐 Chuyến bay **Không delay**.\nXác suất delay: **{prob:.2%}**")

# ==========================================================
# 6️⃣ Footer
# ==========================================================
st.markdown("---")
st.caption("Made by Quang, Thanh — Data Science Final Project ✈️")
