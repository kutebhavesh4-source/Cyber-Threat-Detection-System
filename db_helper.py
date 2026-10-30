import sqlite3
import pandas as pd
import datetime

DB_NAME = "traffic_logs.db"

def init_db():
    """
    Initialize SQLite database and create scan_history table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            destination_port INTEGER,
            flow_duration INTEGER,
            total_fwd_packets INTEGER,
            total_bwd_packets INTEGER,
            packet_length_mean REAL,
            syn_flag_count INTEGER,
            prediction TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_single_log(port, duration, fwd_pack, bwd_pack, pkt_len, syn_flag, prediction, confidence):
    """
    Save manual traffic scan assessment result to SQLite database.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO scan_history 
        (timestamp, destination_port, flow_duration, total_fwd_packets, total_bwd_packets, packet_length_mean, syn_flag_count, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (now, port, duration, fwd_pack, bwd_pack, pkt_len, syn_flag, prediction, confidence))
    
    conn.commit()
    conn.close()

def fetch_scan_history():
    """
    Fetch all historical scan records from SQLite database sorted by latest first.
    """
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM scan_history ORDER BY id DESC", conn)
    conn.close()
    return df
def clear_all_scan_history():
    """
    Delete all records from the scan_history table.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history")
    try:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='scan_history'")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def delete_single_log_by_id(log_id):
    """
    Delete a specific scan record by its unique ID.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM scan_history WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()