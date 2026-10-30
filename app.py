import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from db_helper import (
    clear_all_scan_history,
    delete_single_log_by_id,
    fetch_scan_history,
    init_db,
    save_single_log,
)

# Initialize database connection and create table on application startup
init_db()

# ==========================================
# 1. Page Configuration & Title
# ==========================================
st.set_page_config(page_title="Cyber Threat Detection System", layout="wide")

st.title("⚔️ Cyber Threat Detection System")
st.write("Real-time network traffic classification using Machine Learning.")

# Feature sequence expected by the trained model and scaler
FEATURE_NAMES = [
    'Destination Port',
    'Flow Duration (ms)',
    'Total Forward Packets',
    'Total Backward Packets',
    'Packet Length Mean',
    'SYN Flag Count',
]


# ==========================================
# 2. CSV Preprocessing & Filtering Helper Tool
# ==========================================
def preprocess_real_world_csv(uploaded_df):
  """Maps and filters raw uploaded CSV files (e.g., CIC-IDS2017 or Wireshark logs)

  to extract and rename the 6 required model features automatically.
  """
  # Common column aliases used across different security datasets
  column_mapping = {
      'Dst Port': 'Destination Port',
      'Destination_Port': 'Destination Port',
      'dst_port': 'Destination Port',
      'Flow Duration': 'Flow Duration (ms)',
      'flow_duration': 'Flow Duration (ms)',
      'Total Fwd Packets': 'Total Forward Packets',
      'tot_fwd_pkts': 'Total Forward Packets',
      'Total Backward Packets': 'Total Backward Packets',
      'Tot Bwd Pkts': 'Total Backward Packets',
      'tot_bwd_pkts': 'Total Backward Packets',
      'Pkt Len Mean': 'Packet Length Mean',
      'pkt_len_mean': 'Packet Length Mean',
      'SYN Flag Cnt': 'SYN Flag Count',
      'syn_flag_cnt': 'SYN Flag Count',
  }

  # Rename columns according to standard mapping dictionary
  df_renamed = uploaded_df.rename(columns=column_mapping)

  # Check available model columns in the uploaded file
  available_cols = [col for col in FEATURE_NAMES if col in df_renamed.columns]

  if len(available_cols) == 6:
    filtered_df = df_renamed[FEATURE_NAMES].dropna()
    return filtered_df, True
  else:
    missing = set(FEATURE_NAMES) - set(available_cols)
    return None, f"Missing required feature columns: {', '.join(missing)}"


# ==========================================
# 3. Artifact Loading Functions
# ==========================================
@st.cache_resource
def load_artifacts():
  model = joblib.load('threat_model.pkl')
  scaler = joblib.load('scaler.pkl')
  return model, scaler


try:
  model, scaler = load_artifacts()

  logo_path = 'LOGO_small.png' if os.path.exists('LOGO_small.png') else 'LOGO.png'
  if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=100)

  st.sidebar.success("Model artifacts loaded successfully!")
except Exception as e:
  st.error(f'⚠️ Failed to load model files: {e}')
  st.warning(
      "Please run 'python train_model.py' in your terminal first to generate"
      ' model files.'
  )
  st.stop()

# ==========================================
# 4. Mode Selection & Helper Tools
# ==========================================
mode = st.sidebar.radio(
    'Detection Mode',
    ['Manual Input', 'Batch CSV Log Upload', '📜 Scan History (Database)'],
)

# Downloadable sample CSV for testing
st.sidebar.markdown('---')
st.sidebar.subheader('🧪 Testing Helper')
sample_data = pd.DataFrame({
    'Destination Port': [80, 22, 443, 8080, 21],
    'Flow Duration (ms)': [1500, 50, 3000, 120, 25],
    'Total Forward Packets': [10, 500, 12, 1000, 250],
    'Total Backward Packets': [8, 0, 10, 2, 0],
    'Packet Length Mean': [128.5, 40.0, 512.0, 15.5, 30.0],
    'SYN Flag Count': [0, 1, 0, 1, 1],
})

st.sidebar.download_button(
    label='📥 Download Sample CSV',
    data=sample_data.to_csv(index=False).encode('utf-8'),
    file_name='sample_network_logs.csv',
    mime='text/csv',
)

# ==========================================
# Database Management in Sidebar
# ==========================================
st.sidebar.markdown('---')
st.sidebar.subheader('🗄️ Database Management')
sidebar_history = fetch_scan_history()
total_scans_in_db = len(sidebar_history)
st.sidebar.write(f'Total Saved Scans: **{total_scans_in_db}**')

if total_scans_in_db > 0:
  if st.sidebar.button(
      '🗑️ Clear All Scan History',
      key='sidebar_clear_all_btn',
      use_container_width=True,
  ):
    clear_all_scan_history()
    st.sidebar.success('Scan history cleared successfully!')
    st.rerun()
else:
  st.sidebar.caption('Database is currently empty.')

# ==========================================
# 5. Mode 1: Manual Feature Assessment
# ==========================================
if mode == 'Manual Input':
  st.subheader('Manual Feature Assessment')

  col1, col2, col3 = st.columns(3)
  with col1:
    destination_port = st.number_input(
        'Destination Port', min_value=0, max_value=65535, value=80
    )
    flow_duration = st.number_input(
        'Flow Duration (ms)', min_value=0, value=1500
    )
  with col2:
    total_fwd_packets = st.number_input(
        'Total Forward Packets', min_value=0, value=10
    )
    total_bwd_packets = st.number_input(
        'Total Backward Packets', min_value=0, value=8
    )
  with col3:
    packet_length_mean = st.number_input(
        'Packet Length Mean', min_value=0.0, value=128.5
    )
    syn_flag_count = st.selectbox('SYN Flag Count', [0, 1])

  if st.button('Analyze Traffic Risk', type='primary'):
    raw_features = pd.DataFrame(
        [[
            destination_port,
            flow_duration,
            total_fwd_packets,
            total_bwd_packets,
            packet_length_mean,
            syn_flag_count,
        ]],
        columns=FEATURE_NAMES,
    )

    scaled_features = scaler.transform(raw_features)
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]

    pred_label = 'Malicious' if prediction == 1 else 'Benign'
    conf_score = (
        probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100
    )

    # Automatically persist single scan results to SQLite database
    save_single_log(
        destination_port,
        flow_duration,
        total_fwd_packets,
        total_bwd_packets,
        packet_length_mean,
        syn_flag_count,
        pred_label,
        conf_score,
    )

    st.markdown('---')
    if prediction == 1:
      st.error(f'⚠️ **Alert:** Threat Detected! (Confidence: {conf_score:.2f}%)')
      st.info('💾 Scan result automatically saved to Database!')
    else:
      st.success(
          f'✅ **Normal:** Benign Traffic (Confidence: {conf_score:.2f}%)'
      )
      st.info('💾 Scan result automatically saved to Database!')

# ==========================================
# 6. Mode 2: Batch Log File Upload (With Auto-Cleaning)
# ==========================================
elif mode == 'Batch CSV Log Upload':
  st.subheader('Batch Traffic Log Analysis')
  uploaded_file = st.file_uploader(
      'Upload Network Traffic Log (CSV)', type=['csv']
  )

  if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)
    st.write(
        f'Uploaded Raw File Preview ({raw_data.shape[1]} total columns):'
    )
    st.dataframe(raw_data.head(3), use_container_width=True)

    # Preprocess and filter uploaded file to extract valid feature set
    clean_data, status = preprocess_real_world_csv(raw_data)

    if status is not True:
      st.error(f'❌ {status}')
    else:
      st.success(
          '✅ CSV automatically processed and filtered to 6 target model'
          ' features!'
      )

      if st.button('Run Batch Threat Detection', type='primary'):
        scaled_batch = scaler.transform(clean_data)
        predictions = model.predict(scaled_batch)

        clean_data['Threat_Prediction'] = np.where(
            predictions == 1, 'Malicious', 'Benign'
        )

        st.write('### Analysis Results')
        st.dataframe(clean_data, use_container_width=True)

        # Metric Calculations
        total_logs = len(clean_data)
        total_threats = (predictions == 1).sum()
        threat_percentage = (
            (total_threats / total_logs) * 100 if total_logs > 0 else 0
        )

        st.markdown('---')
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric('Total Logs Processed', total_logs)
        col_m2.metric('Total Threats Identified', int(total_threats))
        col_m3.metric('Threat Level Percentage', f'{threat_percentage:.1f}%')

        # Risk Status Evaluation
        if threat_percentage > 50:
          st.error('🚨 **System Risk Status: CRITICAL RISK DETECTED (>50%)**')
        elif threat_percentage > 20:
          st.warning('⚠️ **System Risk Status: MODERATE RISK DETECTED**')
        else:
          st.success('✅ **System Risk Status: LOW RISK / SECURE**')

        # Interactive Risk Distribution Pie Chart
        st.markdown('### 📊 Traffic Risk Distribution')
        threat_counts = (
            clean_data['Threat_Prediction'].value_counts().reset_index()
        )
        threat_counts.columns = ['Status', 'Count']

        fig = px.pie(
            threat_counts,
            names='Status',
            values='Count',
            title='Benign vs Malicious Traffic Overview',
            color='Status',
            color_discrete_map={'Benign': '#00d26a', 'Malicious': '#ff4b4b'},
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export Analysis Results
        csv_result = clean_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Threat Analysis Report (CSV)',
            data=csv_result,
            file_name='cyber_threat_report.csv',
            mime='text/csv',
        )

# ==========================================
# 7. Mode 3: Database History View & Deletion
# ==========================================
elif mode == '📜 Scan History (Database)':
  st.subheader('📜 Saved Scan History from SQLite Database')
  history_df = fetch_scan_history()

  if not history_df.empty:
    st.write(f'Total Scans Saved: **{len(history_df)}**')
    st.dataframe(history_df, use_container_width=True)

    col_dl, col_del_all = st.columns([1, 1])

    # Download CSV Export
    with col_dl:
      csv_history = history_df.to_csv(index=False).encode('utf-8')
      st.download_button(
          label='📥 Download Database History (CSV)',
          data=csv_history,
          file_name='database_scan_history.csv',
          mime='text/csv',
          use_container_width=True,
      )

    # Clear All Scan History Option
    with col_del_all:
      if st.button(
          '🗑️ Clear All Scan History',
          type='primary',
          key='main_clear_all_btn',
          use_container_width=True,
      ):
        clear_all_scan_history()
        st.success('All scan history cleared successfully!')
        st.rerun()

    st.markdown('---')
    st.write('### 🗑️ Delete Specific Log Record')

    # Select single log record by ID to remove
    log_ids = history_df['id'].tolist()
    selected_id = st.selectbox('Select Log ID to remove:', log_ids)

    if st.button('Delete Selected Log', key='main_del_single_btn'):
      delete_single_log_by_id(selected_id)
      st.success(f'Log ID #{selected_id} deleted successfully!')
      st.rerun()

  else:
    st.info('No scan history found in the database.')