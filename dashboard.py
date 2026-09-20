import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("📊 Attendance Dashboard")

today = datetime.now().strftime("%Y-%m-%d")
file_path = f"attendance/{today}.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.subheader(f"Attendance for {today}")
    st.dataframe(df, use_container_width=True)
    st.metric("Total Present", len(df))
else:
    st.info("No attendance recorded for today yet.")

# Show all historical records
st.subheader("All Records")
all_records = []
if os.path.exists("attendance"):
    for f in os.listdir("attendance"):
        if f.endswith(".csv"):
            all_records.append(pd.read_csv(os.path.join("attendance", f)))

if all_records:
    combined = pd.concat(all_records, ignore_index=True)
    st.dataframe(combined, use_container_width=True)
else:
    st.info("No historical records found.")