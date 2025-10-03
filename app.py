# Streamlit Attendance Pipeline — single-file rebuild with bug fixes & performance polish

import io
import re
import math
import datetime
import time
from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import csv
import streamlit as st
from databricks.sdk import WorkspaceClient
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ------------------------------------------------------------------------------
# CONFIG / CONSTANTS (secrets must be present in Streamlit)
# ------------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="Attendance Data Transformation Pipeline")

DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]
SQL_WAREHOUSE_ID = st.secrets["SQL_WAREHOUSE_ID"]

CATALOG_NAME = "biometrics"
SCHEMA_NAME = "7dxperts"
BRONZE_TABLE = "attendance_table_bronze"
SILVER_TABLE = "attendance_table_silver"
GOLD_TABLE   = "attendance_table_gold"

UPLOAD_FILE_NAME     = "staging_upload_oldapp.csv"
STAGING_VOLUME_PATH  = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/staging/{UPLOAD_FILE_NAME}"

# ------------------------------------------------------------------------------
# STYLES
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    div[data-testid="stSidebar"] {
        width: 25rem !important;
    }
    .main { padding-top: 0rem; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #667eea; margin: 0.5rem 0;
    }
    .chart-container {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin: 1rem 0;
    }
    /* Scope selectboxes lightly without relying on brittle internal classnames */
    div[data-baseweb="select"] > div { background-color: #f8f9fa; }
    .upload-area {
        border: 2px dashed #667eea; border-radius: 10px; padding: 2rem; text-align: center;
        background: rgba(102, 126, 234, 0.05); margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Attendance Data Transformation Pipeline</h1>
    <p>Excel Processing with Interactive Analytics</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# HELPERS (merged from main.py, unchanged behavior, plus resilience)
# ------------------------------------------------------------------------------


def df_to_csv_bytes(df: pd.DataFrame, index: bool = False, header: bool = True, encoding: str = "utf-8") -> bytes:
    """Converts DataFrame to CSV bytes, quoting all fields to handle special characters robustly."""
    return df.to_csv(index=index, header=header, quoting=csv.QUOTE_ALL).encode(encoding)


def upload_csv_bytes_to_volume(ws: WorkspaceClient, csv_bytes: bytes, volume_path: str) -> None:
    """
    Upload CSV bytes to a Databricks Volume; robust to SDK signature differences.
    (Keeps original behavior, with clearer errors)
    """
    if not hasattr(ws, "files") or not hasattr(ws.files, "upload"):
        raise RuntimeError("WorkspaceClient.files.upload not available in this SDK instance")

    # Try keyword
    try:
        ws.files.upload(path=volume_path, content=csv_bytes, overwrite=True)
        return
    except TypeError:
        pass
    except Exception as e:
        st.warning(f"Upload attempt (keyword) failed: {e}")

    # Try positional
    try:
        ws.files.upload(volume_path, csv_bytes, overwrite=True)
        return
    except Exception as e:
        st.warning(f"Upload attempt (positional) failed: {e}")

    # Try minimal positional
    try:
        ws.files.upload(volume_path, csv_bytes)
        return
    except Exception as e:
        raise RuntimeError(f"All upload attempts failed for path={volume_path}. Error: {e}") from e

def _poll_statement_until_ready(ws: WorkspaceClient, resp, timeout_s: int = 60):
    """Poll a Databricks statement until result is available or timeout."""
    sid = None
    start = time.time()
    while getattr(resp, "result", None) is None and getattr(resp, "manifest", None) is None:
        if time.time() - start > timeout_s:
            break
        time.sleep(0.5)
        st.info(f"resp : {resp}")
        sid = getattr(resp, "statement_id", None) or getattr(resp, "id", None)
        # if sid:
        #     break
    if sid and hasattr(ws.statement_execution, "get"):
        try:
            resp = ws.statement_execution.get(statement_id=sid)
        except Exception:
            current_time = time.time()
            if current_time - last_error_time > 5: # Show error max once every 5s
                st.warning(f"Polling status check failed: {e}. Retrying...")
                last_error_time = current_time
    return resp


def copy_into_table(
    ws: WorkspaceClient,
    target_table_fqdn: str,
    staging_path: str,
    warehouse_id: str,
    poll_timeout: int = 60,
) -> Any:
    """
    Execute COPY INTO target_table FROM '<staging_path>' with FORMAT/COPY options.
    Polls until completion or timeout. Returns final response object.
    """
    copy_sql = f"""
    COPY INTO {target_table_fqdn}
    FROM '{staging_path}'
    FILEFORMAT = CSV
    FORMAT_OPTIONS ('header' = 'true')
    COPY_OPTIONS ('mergeSchema' = 'true', 'force' = 'true')
    """
    resp = ws.statement_execution.execute_statement(statement=copy_sql, warehouse_id=warehouse_id)
    resp = _poll_statement_until_ready(ws, resp, timeout_s=poll_timeout)

    status = getattr(resp, "status", None)
    if status and getattr(status, "state", None) and str(getattr(status, "state")).upper() in ("FAILED", "CANCELLED", "CANCELED"):
        raise RuntimeError(f"COPY INTO failed. Status: {status}.")
    return resp


def upload_df_and_load(
    ws: WorkspaceClient,
    df: pd.DataFrame,
    staging_path: str,
    target_table_fqdn: str,
    warehouse_id: str,
    poll_timeout: int = 60,
) -> Any:
    """Convert df -> CSV bytes -> upload -> COPY INTO target table."""
    csv_bytes = df_to_csv_bytes(df, index=False)
    upload_csv_bytes_to_volume(ws, csv_bytes, staging_path)
    resp = copy_into_table(ws, target_table_fqdn, staging_path, warehouse_id, poll_timeout=poll_timeout)
    return resp


@st.cache_resource(show_spinner=False)
def get_workspace() -> WorkspaceClient:
    """Cache the Databricks client across reruns."""
    return WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

def _sanitize_col(col_name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(col_name))
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "col"

def _extract_rows_from_statement(ws: WorkspaceClient, stmt_resp) -> List[List[Any]]:
    """Normalize Databricks row results into a list of lists/dicts safely."""
    stmt_resp = _poll_statement_until_ready(ws, stmt_resp, timeout_s=8)
    result_obj = getattr(stmt_resp, "result", None)
    arr_inner = getattr(result_obj, "data_array", None) if result_obj is not None else None
    rows_inner: List[List[Any]] = []
    if arr_inner:
        for rr in arr_inner:
            if hasattr(rr, "as_dict"):
                rows_inner.append(rr.as_dict())
            elif isinstance(rr, dict):
                rows_inner.append(rr)
            elif isinstance(rr, (list, tuple)):
                rows_inner.append(list(rr))
            else:
                vals = getattr(rr, "values", None)
                if vals is not None:
                    rows_inner.append(list(vals))
                else:
                    try:
                        rows_inner.append(dict(rr))
                    except Exception:
                        rows_inner.append([str(rr)])
    return rows_inner

def _fetch_gold_to_df(ws: WorkspaceClient, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> pd.DataFrame:

    """Paginate through GOLD table and build a DataFrame with best-effort column naming."""
    page_size = 6000
    offset = 0
    all_rows: List[Any] = []
    last_resp = None

    # --- ADD THIS BLOCK ---
    # Build a WHERE clause if dates are provided to filter the query
    where_clause = ""
    if start_date and end_date:
        where_clause = f"WHERE work_date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'"
    # --- END OF BLOCK TO ADD ---

    for page_num in range(1000):
        page_sql = (
            f"SELECT * FROM `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{GOLD_TABLE}` "
            f"{where_clause} "  # <-- This part is new
            f"ORDER BY Employee_No, work_date LIMIT {page_size} OFFSET {offset}"
        )
        resp = ws.statement_execution.execute_statement(statement=page_sql, warehouse_id=SQL_WAREHOUSE_ID)
        resp = _poll_statement_until_ready(ws, resp, timeout_s=30)
        last_resp = resp

        result_obj = getattr(resp, "result", None)
        arr = getattr(result_obj, "data_array", None) if result_obj is not None else None

        rows_page: List[Any] = []
        if arr:
            for r in arr:
                if hasattr(r, "as_dict"):
                    rows_page.append(r.as_dict()); continue
                if isinstance(r, dict):
                    rows_page.append(r); continue
                vals = getattr(r, "values", None)
                if vals is not None:
                    rows_page.append(list(vals)); continue
                if isinstance(r, (list, tuple)):
                    rows_page.append(list(r)); continue
                try:
                    rows_page.append(dict(r))
                except Exception:
                    rows_page.append({"raw_row": str(r)})
        else:
            if page_num == 0:
                st.write("No rows returned for Gold table (it may be empty).")
            break

        all_rows.extend(rows_page)
        fetched = len(rows_page)
        st.write(f"Fetched page {page_num+1}: {fetched} rows (offset {offset})")
        offset += page_size
        if fetched < page_size:
            break

    if not all_rows:
        return pd.DataFrame()

    # If first element is positional, fetch column names
    if isinstance(all_rows[0], list):
        cols = None
        # SHOW COLUMNS
        try:
            show_sql = f"SHOW COLUMNS IN `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{GOLD_TABLE}`"
            show_resp = ws.statement_execution.execute_statement(statement=show_sql, warehouse_id=SQL_WAREHOUSE_ID)
            show_rows = _extract_rows_from_statement(ws, show_resp)
            if show_rows:
                candidate = []
                for r in show_rows:
                    if isinstance(r, dict) and len(r) > 0:
                        candidate.append(list(r.values())[0])
                    elif isinstance(r, (list, tuple)) and len(r) > 0:
                        candidate.append(r[0])
                    else:
                        candidate.append(str(r))
                if candidate:
                    cols = candidate
        except Exception:
            cols = None

        # Fallback: DESCRIBE
        if not cols:
            try:
                desc_sql = f"DESCRIBE `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{GOLD_TABLE}`"
                desc_resp = ws.statement_execution.execute_statement(statement=desc_sql, warehouse_id=SQL_WAREHOUSE_ID)
                desc_rows = _extract_rows_from_statement(ws, desc_resp)
                if desc_rows:
                    candidate = []
                    for r in desc_rows:
                        if isinstance(r, dict) and len(r) > 0:
                            candidate.append(list(r.values())[0])
                        elif isinstance(r, (list, tuple)) and len(r) > 0:
                            candidate.append(r[0])
                        else:
                            candidate.append(str(r))
                    if candidate:
                        cols = candidate
            except Exception:
                cols = None

        # If lengths match, great; otherwise best-effort
        if cols and len(cols) == len(all_rows[0]):
            return pd.DataFrame(all_rows, columns=cols)
        else:
            # Try manifest-based schema
            try:
                manifest = getattr(last_resp, "manifest", None)
                if manifest is not None:
                    schema = getattr(manifest, "schema", None)
                    if schema and getattr(schema, "columns", None):
                        cols_from_manifest = [c.name for c in schema.columns]
                        if len(cols_from_manifest) == len(all_rows[0]):
                            return pd.DataFrame(all_rows, columns=cols_from_manifest)
                        else:
                            return pd.DataFrame(all_rows, columns=cols_from_manifest[:len(all_rows[0])])
                    else:
                        return pd.DataFrame(all_rows)
                else:
                    if cols:
                        if len(cols) < len(all_rows[0]):
                            extended = cols + [f"col_{i}" for i in range(len(cols), len(all_rows[0]))]
                            return pd.DataFrame(all_rows, columns=extended)
                        else:
                            return pd.DataFrame(all_rows, columns=cols[:len(all_rows[0])])
                    else:
                        return pd.DataFrame(all_rows, columns=[f"col_{i}" for i in range(len(all_rows[0]))])
            except Exception:
                return pd.DataFrame(all_rows)

    # Dict-like rows already
    try:
        return pd.DataFrame(all_rows)
    except Exception:
        return pd.DataFrame(all_rows)

# ------------------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------------------
for k, v in {
    "pipeline_ran": False,
    "gold_df": pd.DataFrame(),
    "excel_bytes": None,
    "data_fetch_completed": False,
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ------------------------------------------------------------------------------
# UI — Upload / Actions
# ------------------------------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Upload File")
    uploaded_files = st.file_uploader("Choose Employee Swipe Details Excel files", type=["xlsx"], accept_multiple_files=True, label_visibility="collapsed")

with col2:
    st.markdown("### Actions")
    if st.button("Clear Pipeline Cache", use_container_width=True):
        for key in ["pipeline_ran", "gold_df", "excel_bytes", "data_fetch_completed"]:
            st.session_state[key] = (pd.DataFrame() if key == "gold_df" else (False if key in ("pipeline_ran","data_fetch_completed") else None))
        st.success("Cache cleared!")
        st.rerun()

# ------------------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------------------
def _get_table_count(ws: WorkspaceClient, table_fqdn: str, warehouse_id: str, timeout_s: int = 20) -> int:
    """Run SELECT COUNT(*) on a table and return integer count (uses _extract_rows_from_statement helper)."""
    count_sql = f"SELECT COUNT(*) AS cnt FROM {table_fqdn}"
    resp = ws.statement_execution.execute_statement(statement=count_sql, warehouse_id=warehouse_id)
    resp = _poll_statement_until_ready(ws, resp, timeout_s=timeout_s)
    rows = _extract_rows_from_statement(ws, resp)
    if not rows:
        return 0
    # rows could be list of dicts or list; handle common cases
    first = rows[0]
    if isinstance(first, dict):
        # value may be under 'cnt' or similar
        for k in first:
            try:
                return int(first[k])
            except Exception:
                continue
        return 0
    if isinstance(first, (list, tuple)):
        try:
            return int(first[0])
        except Exception:
            return 0
    try:
        return int(first)
    except Exception:
        return 0

def run_full_pipeline(file: UploadedFile) -> Tuple[pd.DataFrame, bytes]:
    ws = get_workspace()
    st.info("Reading uploaded Excel into pandas...")
    try:
        df_bronze = pd.read_excel(file, sheet_name="mainsheet")
    except Exception:
        # fallback: read first sheet
        file.seek(0)
        df_bronze = pd.read_excel(file, sheet_name=0)

    # --- ROBUST DATA CLEANING BLOCK ---
    st.info("Performing data cleaning on DataFrame...")
    initial_rows = len(df_bronze)

    # 1. Drop rows where 'Employee No' is entirely missing.
    df_bronze.dropna(subset=['Employee No'], inplace=True)

    # 2. Convert 'Employee No' to a string type to handle mixed formats safely.
    df_bronze['Employee No'] = df_bronze['Employee No'].astype(str).str.strip()

    df_bronze = df_bronze[df_bronze['Employee No'].str.contains(r'[a-zA-Z0-9]')]

    final_rows = len(df_bronze)
    removed_count = initial_rows - final_rows
    if removed_count > 0:
        st.warning(f"Data Cleaning: Removed {removed_count} problematic or empty rows.")

    # --- END OF CLEANING BLOCK ---

    # Determine the date range from the uploaded file to filter the final output
    df_bronze['Date'] = pd.to_datetime(df_bronze['Swipe Date']).dt.date
    start_date = df_bronze['Date'].min()
    end_date = df_bronze['Date'].max()
    st.info(f"Detected date range in uploaded file: {start_date} to {end_date}")
    # --- END OF FIX ---

    # show quick preview so user can confirm before we upload
    st.write("Preview of parsed dataframe (first 5 rows):")
    st.write(f"shape: {df_bronze.shape}")
    st.dataframe(df_bronze.head(5))


    # derive date parts if missing (keeps your Silver SQL unchanged)
    try:
        df_bronze['Date'] = pd.to_datetime(df_bronze['Swipe Date']).dt.date
        df_bronze['time'] = pd.to_datetime(df_bronze['Swipe Date']).dt.time
        df_bronze['Year'] = pd.to_datetime(df_bronze['Swipe Date']).dt.year
        df_bronze['Day']  = pd.to_datetime(df_bronze['Swipe Date']).dt.day
        df_bronze['Month_short_name'] = pd.to_datetime(df_bronze['Swipe Date']).dt.strftime('%b')
        df_bronze['Month_Number'] = pd.to_datetime(df_bronze['Swipe Date']).dt.month

    except Exception as e:
        st.warning("Failed to auto-derive date/time columns from 'Swipe Date' — continuing but Silver may fail.")
        st.exception(e)

    # Sanitize / map columns (existing manual map)
    manual_col_map = {
        "Employee No": "Employee_No",
        "Employee Name": "Employee_Name",
        "Access Card": "Access_Card",
        "Shift": "Shift",
        "Swipe Date": "Swipe_Date",
        "Door/Address": "Door_Address",
        "In/Out": "In_Out",
        "Longitude": "Longitude",
        "Latitude": "Latitude",
        "Location Type": "Location_Type",
        "Mobile Device Name": "Mobile_Device_Name",
        "Mobile Device Id": "Mobile_Device_Id",
        "Swipe Type": "Swipe_Type",
        "Status": "Status",
        "Remarks": "Remarks",
        "Permission Reason": "Permission_Reason",
        "Signed In/Out By": "Signed_In_Out_By",
        "Received On": "Received_On",
    }

    col_map = {orig: manual_col_map.get(orig, _sanitize_col(orig)) for orig in df_bronze.columns}
    df_for_upload = df_bronze.rename(columns=col_map)
    
    # final quick sanity
    st.write("Columns that will be uploaded:")
    st.write(list(df_for_upload.columns))
    st.write("Rows to upload:", len(df_for_upload))

    # Create Bronze table structure if it doesn't exist
    st.info("Ensuring Bronze table exists and merging uploaded rows...")

    def _pandas_dtype_to_sql(dtype) -> str:
        import pandas as _pd
        if _pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        if _pd.api.types.is_float_dtype(dtype):
            return "DOUBLE"
        if _pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        if _pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        return "STRING"

    bronze_cols = list(df_for_upload.columns)
    bronze_cols_sanitized = [_sanitize_col(c) for c in bronze_cols]

    col_defs = []
    for original_col, safe_col in zip(bronze_cols, bronze_cols_sanitized):
        sql_type = _pandas_dtype_to_sql(df_for_upload[original_col].dtype)
        col_defs.append(f"`{safe_col}` {sql_type}")

    bronze_table_fqdn = f"`{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{BRONZE_TABLE}`"
    cols_text = ",\n  ".join(col_defs)

    create_bronze_ddl = f"""CREATE TABLE IF NOT EXISTS {bronze_table_fqdn} (
    {cols_text}
    )"""

    ws.statement_execution.execute_statement(statement=create_bronze_ddl, warehouse_id=SQL_WAREHOUSE_ID)
    _poll_statement_until_ready(ws, ws.statement_execution.execute_statement(statement=f"SELECT 1 FROM {bronze_table_fqdn} LIMIT 1", warehouse_id=SQL_WAREHOUSE_ID), timeout_s=20)

    # Prepare DataFrame with sanitized column names
    rename_map = {orig: _sanitize_col(orig) for orig in bronze_cols}
    df_to_merge = df_for_upload.rename(columns=rename_map).copy()

    # 1. Convert DataFrame to CSV WITHOUT a header row
    st.info("Uploading data to staging volume for efficient MERGE...")
    csv_bytes = df_to_csv_bytes(df_to_merge, header=False, index=False)
    upload_csv_bytes_to_volume(ws, csv_bytes, STAGING_VOLUME_PATH)
    st.success("✅ Staging file uploaded.")

    # 2. Explicitly define the schema to avoid any parsing ambiguity
    schema_list = []
    for col in df_to_merge.columns:
        # Re-use the helper function to map pandas dtypes to SQL types
        sql_type = _pandas_dtype_to_sql(df_to_merge[col].dtype)
        schema_list.append(f"`{col}` {sql_type}")
    schema_sql = ", ".join(schema_list)

    update_clauses = [f"tgt.`{col}` = src.`{col}`" for col in df_for_upload.columns]
    update_set_clause = ",\n    ".join(update_clauses)
    insert_cols = ", ".join([f"`{col}`" for col in df_for_upload.columns])
    insert_vals = ", ".join([f"src.`{col}`" for col in df_for_upload.columns])

    update_condition_cols = [col for col in df_for_upload.columns if col not in ['Employee_No', 'Swipe_Date']]
    update_condition = " OR ".join([f"tgt.`{col}` IS DISTINCT FROM src.`{col}`" for col in update_condition_cols])

    # 4. Construct the efficient AND conditional MERGE statement
    merge_bronze_sql = f"""
    MERGE INTO {bronze_table_fqdn} AS tgt
    USING (
        SELECT * FROM READ_FILES('{STAGING_VOLUME_PATH}', format => 'csv', schema => '{schema_sql}', multiLine => true)
    ) AS src
    ON tgt.`Employee_No` = src.`Employee_No` AND tgt.`Swipe_Date` = src.`Swipe_Date`
    WHEN MATCHED AND ({update_condition}) THEN
        UPDATE SET {update_set_clause}
    WHEN NOT MATCHED THEN
        INSERT ({insert_cols}) VALUES ({insert_vals})
    """

    # 5. Execute the efficient MERGE statement (this is unchanged)
    try:
        st.info("Executing optimized MERGE into Bronze table...")
        merge_resp = ws.statement_execution.execute_statement(statement=merge_bronze_sql, warehouse_id=SQL_WAREHOUSE_ID, wait_timeout='50s')
        _poll_statement_until_ready(ws, merge_resp, timeout_s=0)
        st.success(f"✅ Bronze table merged successfully using staged file.")
    except Exception as e:
        st.error("Failed merging into Bronze table using the efficient file-based method.")
        st.exception(e)
        raise

    st.success("✅ Bronze table loaded successfully.")

    # Create Silver
    st.info("Creating Silver table (cleaning & casting) ...")

    silver_source_query = f"""
        SELECT
            CAST(`Employee_No` AS STRING) AS Employee_No, 
            CAST(`Employee_Name` AS STRING) AS Employee_Name,
            regexp_replace(`Access_Card`, '[^0-9]', '') AS Access_Card, 
            CAST(`Shift` AS STRING) AS Shift,
            CAST(`Swipe_Date` AS STRING) AS Swipe_Date,
            CAST(`Year` AS INT) AS Year,
            CAST(`Day` AS INT) AS Day,
            CAST(`Month_short_name` AS STRING) AS Month_short_name,
            CAST(`Month_Number` AS INT) AS Month_Number,
            CAST(`Date` AS date) AS Date,
            CAST(`time` AS timestamp) AS time,
            CAST(`Door_Address` AS STRING) AS Door_Address,
            CAST(`In_Out` AS STRING) AS In_Out,
            CAST(`Longitude` AS DOUBLE) AS Longitude,
            CAST(`Latitude` AS DOUBLE) AS Latitude,
            CAST(`Location_Type` AS STRING) AS Location_Type,
            CAST(`Mobile_Device_Name` AS STRING) AS Mobile_Device_Name,
            CAST(`Mobile_Device_Id` AS STRING) AS Mobile_Device_Id,
            CAST(`Swipe_Type` AS STRING) AS Swipe_Type,
            CAST(`Status` AS STRING) AS Status,
            CAST(`Remarks` AS STRING) AS Remarks,
            CAST(`Permission_Reason` AS STRING) AS Permission_Reason,
            CAST(`Signed_In_Out_By` AS STRING) AS Signed_In_Out_By,
            CAST(`Received_On` AS timestamp) AS Received_On
        FROM `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{BRONZE_TABLE}`
    """

    # Define the columns for the conditional update (all columns except the join keys)
    silver_cols = [
        'Employee_Name', 'Access_Card', 'Shift', 'Year', 'Day', 'Month_short_name',
        'Month_Number', 'Date', 'time', 'Door_Address', 'In_Out', 'Longitude', 'Latitude',
        'Location_Type', 'Mobile_Device_Name', 'Mobile_Device_Id', 'Swipe_Type', 'Status',
        'Remarks', 'Permission_Reason', 'Signed_In_Out_By', 'Received_On'
    ]
    silver_update_condition = " OR ".join([f"tgt.`{col}` IS DISTINCT FROM src.`{col}`" for col in silver_cols])

    # The final, optimized MERGE statement for the Silver table
    merge_silver_sql = f"""
    MERGE INTO `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{SILVER_TABLE}` AS tgt
    USING (
    SELECT * FROM (
        SELECT
        CAST(`Employee_No` AS STRING) AS Employee_No,
        CAST(`Employee_Name` AS STRING) AS Employee_Name,
        regexp_replace(`Access_Card`, '[^0-9]', '') AS Access_Card,
        CAST(`Shift` AS STRING) AS Shift,
        CAST(`Swipe_Date` AS STRING) AS Swipe_Date,
        CAST(`Year` AS INT) AS Year,
        CAST(`Day` AS INT) AS Day,
        CAST(`Month_short_name` AS STRING) AS Month_short_name,
        CAST(`Month_Number` AS INT) AS Month_Number,
        CAST(`Date` AS DATE) AS Date,
        CAST(`time` AS TIMESTAMP) AS time,
        CAST(`Door_Address` AS STRING) AS Door_Address,
        CAST(`In_Out` AS STRING) AS In_Out,
        CAST(`Longitude` AS DOUBLE) AS Longitude,
        CAST(`Latitude` AS DOUBLE) AS Latitude,
        CAST(`Location_Type` AS STRING) AS Location_Type,
        CAST(`Mobile_Device_Name` AS STRING) AS Mobile_Device_Name,
        CAST(`Mobile_Device_Id` AS STRING) AS Mobile_Device_Id,
        CAST(`Swipe_Type` AS STRING) AS Swipe_Type,
        CAST(`Status` AS STRING) AS Status,
        CAST(`Remarks` AS STRING) AS Remarks,
        CAST(`Permission_Reason` AS STRING) AS Permission_Reason,
        CAST(`Signed_In_Out_By` AS STRING) AS Signed_In_Out_By,
        CAST(`Received_On` AS TIMESTAMP) AS Received_On,
        ROW_NUMBER() OVER (PARTITION BY `Employee_No`, `Swipe_Date` ORDER BY `Received_On` DESC) AS rn
        FROM `biometrics`.`7dxperts`.`attendance_table_bronze`
        WHERE Date BETWEEN '{start_date}' AND '{end_date}'
    ) WHERE rn = 1
    ) AS src
    ON tgt.Employee_No = src.Employee_No AND tgt.Swipe_Date = src.Swipe_Date
    WHEN MATCHED AND ({silver_update_condition}) THEN
        UPDATE SET *
    WHEN NOT MATCHED THEN
        INSERT *
    """

    create_silver_if_missing_sql = f"""
    CREATE TABLE IF NOT EXISTS `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{SILVER_TABLE}` AS
    {silver_source_query} LIMIT 0
"""

    # Execute the statements in order
    ws.statement_execution.execute_statement(statement=create_silver_if_missing_sql, warehouse_id=SQL_WAREHOUSE_ID)
    silver_merge_resp = ws.statement_execution.execute_statement(statement=merge_silver_sql, warehouse_id=SQL_WAREHOUSE_ID, wait_timeout='50s')
    _poll_statement_until_ready(ws, silver_merge_resp, timeout_s=0) # Wait for MERGE to finish
    st.success("✅ Silver table merged (upserted) from Bronze.")

    # Create Gold
    st.info("Merging into Gold table (business logic) ...")

    # Update constants
    STANDARD_WORK_HOURS = 9  # 9 hours as per HR requirement
    STANDARD_WORK_SECONDS = STANDARD_WORK_HOURS * 3600  # 32400 seconds

    # Replace the existing create_gold_table_sql with this one
    create_gold_table_sql = f"""
CREATE TABLE IF NOT EXISTS `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{GOLD_TABLE}` (
    Employee_No STRING,
    Employee_Name STRING,
    work_date DATE,
    sign_in_method STRING,
    swipe_status STRING,
    in_time STRING,
    out_time STRING,
    in_duration STRING,
    out_duration STRING,
    full_span STRING,
    Productive_Overtime STRING,
    Productive_Shortfall STRING,
    Overall_Overtime STRING,
    Overall_Shortfall STRING,
    in_time_minutes DOUBLE,
    out_time_minutes DOUBLE,
    in_duration_minutes DOUBLE,
    out_duration_minutes DOUBLE,
    Productive_Overtime_Minutes DOUBLE,
    Productive_Shortfall_Minutes DOUBLE,
    Overall_Overtime_Minutes DOUBLE,
    Overall_Shortfall_Minutes DOUBLE
)
"""

    create_resp = ws.statement_execution.execute_statement(statement=create_gold_table_sql, warehouse_id=SQL_WAREHOUSE_ID, wait_timeout='50s')
    _poll_statement_until_ready(ws, create_resp, timeout_s=0)
    st.success("✅ Gold table schema created/verified.")

    gold_data_query = f"""
WITH swipes AS (
    SELECT
        Employee_No, Employee_Name,
        to_timestamp(concat(date_format(to_date(`Date`), 'yyyy-MM-dd'), ' ', coalesce(date_format(`time`, 'HH:mm:ss'), cast(`time` as string))), 'yyyy-MM-dd HH:mm:ss') AS swipe_ts,
        CASE WHEN HOUR(to_timestamp(concat(date_format(to_date(`Date`), 'yyyy-MM-dd'), ' ', coalesce(date_format(`time`, 'HH:mm:ss'), cast(`time` as string))), 'yyyy-MM-dd HH:mm:ss')) < 5
             THEN date_sub(to_date(`Date`), 1) ELSE to_date(`Date`) END AS work_date,
        UPPER(trim(`In_Out`)) AS io, `Swipe_Type`
    FROM `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{SILVER_TABLE}`
    WHERE `Date` BETWEEN '{start_date}' AND '{end_date}'
    AND Status = 'Approved' AND `time` IS NOT NULL AND `In_Out` IN ('In', 'Out', 'IN', 'OUT')

),
first_swipe_details AS (
    SELECT Employee_No, work_date, Swipe_Type as first_swipe_type
    FROM (SELECT *, ROW_NUMBER() OVER(PARTITION BY Employee_No, work_date ORDER BY swipe_ts ASC) as rn FROM swipes)
    WHERE rn = 1
),
grouped_consecutive AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY Employee_No, work_date, io ORDER BY swipe_ts) AS rn_io,
        ROW_NUMBER() OVER (PARTITION BY Employee_No, work_date ORDER BY swipe_ts) AS rn_all
    FROM swipes
),
groups AS (SELECT *, rn_all - rn_io AS grp FROM grouped_consecutive),
collapsed AS (
    SELECT Employee_No, Employee_Name, work_date, io, MAX(swipe_ts) AS effective_ts
    FROM groups GROUP BY Employee_No, Employee_Name, work_date, io, grp
),
ordered_collapsed AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY Employee_No, work_date ORDER BY effective_ts) AS seq FROM collapsed
),
windows AS (
    SELECT *,
        FIRST_VALUE(io) OVER (PARTITION BY Employee_No, work_date ORDER BY seq) AS first_io_temp,
        LAST_VALUE(io) OVER (PARTITION BY Employee_No, work_date ORDER BY seq ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_io_temp
    FROM ordered_collapsed
),
first_last AS (
    SELECT Employee_No, MAX(Employee_Name) AS Employee_Name, work_date, MAX(first_io_temp) AS first_io, MAX(last_io_temp) AS last_io,
           MIN(CASE WHEN io = 'IN' THEN effective_ts END) AS min_in_ts, MAX(CASE WHEN io = 'OUT' THEN effective_ts END) AS max_out_ts
    FROM windows GROUP BY Employee_No, work_date
),
pairs AS (
    SELECT *,
        LEAD(effective_ts) OVER (PARTITION BY Employee_No, work_date ORDER BY seq) AS end_ts,
        LEAD(io) OVER (PARTITION BY Employee_No, work_date ORDER BY seq) AS end_io,
        LAG(io) OVER (PARTITION BY Employee_No, work_date ORDER BY seq) AS prev_io
    FROM ordered_collapsed
),
in_durs AS (
    SELECT Employee_No, work_date, TIMESTAMPDIFF(SECOND, effective_ts, end_ts) AS dur_seconds
    FROM pairs WHERE io = 'IN' AND end_io = 'OUT' AND TIMESTAMPDIFF(SECOND, effective_ts, end_ts) > 0
),
out_durs AS (
    SELECT Employee_No, work_date, TIMESTAMPDIFF(SECOND, effective_ts, end_ts) AS dur_seconds
    FROM pairs WHERE io = 'OUT' AND end_io = 'IN' AND prev_io = 'IN' AND TIMESTAMPDIFF(SECOND, effective_ts, end_ts) > 0
),
agg_in AS (SELECT Employee_No, work_date, SUM(dur_seconds) AS in_seconds FROM in_durs GROUP BY Employee_No, work_date),
agg_out AS (SELECT Employee_No, work_date, SUM(dur_seconds) AS out_seconds FROM out_durs GROUP BY Employee_No, work_date),
final_calculations AS (
    SELECT
        f.Employee_No, f.Employee_Name, f.work_date, f.first_io, f.last_io, f.min_in_ts, f.max_out_ts,
        fsd.first_swipe_type,
        COALESCE(a.in_seconds, 0) AS in_seconds,
        COALESCE(o.out_seconds, 0) AS out_seconds,
        (CASE WHEN f.first_io <> 'OUT' AND f.last_io <> 'IN' THEN TIMESTAMPDIFF(SECOND, f.min_in_ts, f.max_out_ts)
              ELSE COALESCE(a.in_seconds, 0) + COALESCE(o.out_seconds, 0)
         END) AS raw_full_span_seconds
    FROM first_last f
    LEFT JOIN agg_in a ON f.Employee_No = a.Employee_No AND f.work_date = a.work_date
    LEFT JOIN agg_out o ON f.Employee_No = o.Employee_No AND f.work_date = o.work_date
    LEFT JOIN first_swipe_details fsd ON f.Employee_No = fsd.Employee_No AND f.work_date = fsd.work_date
    WHERE (f.min_in_ts IS NOT NULL OR f.max_out_ts IS NOT NULL)
)
SELECT
    Employee_No,
    Employee_Name,
    work_date,
    CASE WHEN first_swipe_type LIKE '%Web%' THEN 'Web Sign-In' ELSE 'Office Sign-In' END as sign_in_method,
    CASE WHEN first_io <> 'OUT' AND last_io <> 'IN' THEN 'Complete swipe' ELSE 'Incomplete swipe' END AS swipe_status,
    CASE WHEN first_io = 'OUT' THEN '' ELSE CAST(min_in_ts AS STRING) END AS in_time,
    CASE WHEN last_io = 'IN' THEN '' ELSE CAST(max_out_ts AS STRING) END AS out_time,
    
    -- In Duration (with Web Sign-In cap)
    CONCAT(LPAD(FLOOR(CEIL((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) / 60.0) / 60), 2, '0'), ':', LPAD(CEIL((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) / 60.0) % 60, 2, '0')) AS in_duration,
    CONCAT(LPAD(FLOOR(CEIL(out_seconds / 60.0) / 60), 2, '0'), ':', LPAD(CEIL(out_seconds / 60.0) % 60, 2, '0')) AS out_duration,
    CONCAT(LPAD(FLOOR(CEIL((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) / 60.0) / 60), 2, '0'), ':', LPAD(CEIL((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) / 60.0) % 60, 2, '0')) AS full_span,
    
    -- Productive Overtime (8-hour standard, using capped in_duration)
    CONCAT(LPAD(FLOOR(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) > 28800 THEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) - 28800 ELSE 0 END / 60.0) / 60), 2, '0'), ':', LPAD(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) > 28800 THEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) - 28800 ELSE 0 END / 60.0) % 60, 2, '0')) AS Productive_Overtime,
    
    -- Productive Shortfall (8-hour standard, using capped in_duration)
    CONCAT(LPAD(FLOOR(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) < 28800 THEN 28800 - (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) ELSE 0 END / 60.0) / 60), 2, '0'), ':', LPAD(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) < 28800 THEN 28800 - (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) ELSE 0 END / 60.0) % 60, 2, '0')) AS Productive_Shortfall,
    
    -- Overall Overtime (9-hour standard)
    CONCAT(LPAD(FLOOR(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) > 32400 THEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) - 32400 ELSE 0 END / 60.0) / 60), 2, '0'), ':', LPAD(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) > 32400 THEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) - 32400 ELSE 0 END / 60.0) % 60, 2, '0')) AS Overall_Overtime,
    
    -- Overall Shortfall (9-hour standard)
    CONCAT(LPAD(FLOOR(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) < 32400 THEN 32400 - (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) ELSE 0 END / 60.0) / 60), 2, '0'), ':', LPAD(CEIL(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) < 32400 THEN 32400 - (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) ELSE 0 END / 60.0) % 60, 2, '0')) AS Overall_Shortfall,

    -- Numerical Minutes Columns
    ROUND(CASE WHEN first_io = 'OUT' THEN 0.0 ELSE HOUR(min_in_ts) * 60 + MINUTE(min_in_ts) + SECOND(min_in_ts) / 60.0 END, 0) AS in_time_minutes,
    ROUND(CASE WHEN last_io = 'IN' THEN 0.0 ELSE HOUR(max_out_ts) * 60 + MINUTE(max_out_ts) + SECOND(max_out_ts) / 60.0 END, 0) AS out_time_minutes,
    ROUND((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) / 60.0, 0) AS in_duration_minutes,
    ROUND(out_seconds / 60.0, 0) AS out_duration_minutes,
    ROUND(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) > 28800 THEN ((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) - 28800) / 60.0 ELSE 0.0 END, 0) AS Productive_Overtime_Minutes,
    ROUND(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END) < 28800 THEN (28800 - (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(in_seconds, 32400) ELSE in_seconds END)) / 60.0 ELSE 0.0 END, 0) AS Productive_Shortfall_Minutes,
    ROUND(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) > 32400 THEN ((CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) - 32400) / 60.0 ELSE 0.0 END, 0) AS Overall_Overtime_Minutes,
    ROUND(CASE WHEN (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END) < 32400 THEN (32400 - (CASE WHEN first_swipe_type LIKE '%Web%' THEN LEAST(raw_full_span_seconds, 32400) ELSE raw_full_span_seconds END)) / 60.0 ELSE 0.0 END, 0) AS Overall_Shortfall_Minutes
    FROM final_calculations
"""

    gold_cols = [
    'Employee_Name', 'work_date', 'sign_in_method', 'swipe_status', 'in_time',
    'out_time', 'in_duration', 'out_duration', 'full_span',
    'Productive_Overtime', 'Productive_Shortfall', 'Overall_Overtime', 'Overall_Shortfall',
    'in_time_minutes', 'out_time_minutes', 'in_duration_minutes', 'out_duration_minutes',
    'Productive_Overtime_Minutes', 'Productive_Shortfall_Minutes',
    'Overall_Overtime_Minutes', 'Overall_Shortfall_Minutes'
]
    gold_update_condition = " OR ".join([f"tgt.`{col}` IS DISTINCT FROM src.`{col}`" for col in gold_cols])

    # The date filter is already included in your gold_data_query
    merge_gold_sql = f"""
    MERGE INTO `{CATALOG_NAME}`.`{SCHEMA_NAME}`.`{GOLD_TABLE}` AS tgt
    USING ({gold_data_query}) AS src
    ON tgt.Employee_No = src.Employee_No AND tgt.work_date = src.work_date
    WHEN MATCHED AND ({gold_update_condition}) THEN
        UPDATE SET *
    WHEN NOT MATCHED THEN
        INSERT *
    """

    st.info("Merging processed attendance data into Gold table...")
    merge_resp = ws.statement_execution.execute_statement(statement=merge_gold_sql, warehouse_id=SQL_WAREHOUSE_ID)
    _poll_statement_until_ready(ws, merge_resp, timeout_s=300)

    st.success("✅ Gold table merged successfully with processed attendance data.")
    st.info("Fetching merged GOLD rows into a DataFrame...")
    result_df = _fetch_gold_to_df(ws, start_date=start_date, end_date=end_date)
    st.write("Total rows in GOLD (fetched):", len(result_df))
    # --- NEW: Convert minutes columns to a clean integer type ---
    minutes_cols = [
        'in_time_minutes', 'out_time_minutes', 'in_duration_minutes',
        'out_duration_minutes', 'Productive_Overtime_Minutes',
        'Productive_Shortfall_Minutes', 'Overall_Overtime_Minutes', 
        'Overall_Shortfall_Minutes'
    ]

    for col in minutes_cols:
        if col in result_df.columns:
            # Use pd.to_numeric to handle any potential errors gracefully
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
    # --- END OF NEW CODE ---

    # --- Generate Final CSV Output ---
    csv_bytes = result_df.to_csv(index=False).encode('utf-8-sig')

    return result_df, csv_bytes

    # Download UX
if uploaded_files:
    st.success(f"{len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''} ready for processing: {', '.join(file.name for file in uploaded_files)}")
    run_now = st.button("▶️ Run Full Pipeline", type="primary")
    if run_now:
        with st.spinner("Processing multiple files... This may take a few minutes."):
            try:
                output_files = []  # List to collect (name, bytes) tuples
                for file in uploaded_files:
                    # The return variable is now csv_bytes for clarity
                    result_df, csv_bytes = run_full_pipeline(file) 
                    
                    start_date_str = result_df['work_date'].min() if not result_df.empty else datetime.date.today().strftime('%Y-%m-%d')
                    start_date = pd.to_datetime(start_date_str).strftime('%Y-%m-%d')
                    
                    # The output name now correctly uses the .csv extension
                    output_name = f"attendance_report_{start_date}.csv" 
                    
                    output_files.append((output_name, csv_bytes))
                    st.success(f"Processed '{file.name}' successfully.")

                st.session_state["output_files"] = output_files
                st.session_state["pipeline_ran"] = True
            except Exception as e:
                st.error("An error occurred during multi-file processing:")
                st.exception(e)

    # Download UX (The download button itself was already correct)
    if st.session_state.get("pipeline_ran", False) and "output_files" in st.session_state:
        st.markdown("### Download Processed Reports")
        for name, bytes_data in st.session_state["output_files"]:
            st.download_button(
                label=f"📈 Download {name}",
                data=bytes_data,
                file_name=name,
                mime="text/csv",
                key=f"download_{name}",
            )
    if not run_now:
        st.info("Using previously processed multi-file results.")

    if st.session_state.get("pipeline_ran", False) and not run_now:
        st.info("Using previously run pipeline result.")
        
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 12px; margin: 2rem 0;">
        <h2>Welcome to Enhanced Attendance Pipeline</h2>
        <p style="font-size: 1.1rem; color: #666; margin: 1rem 0;">Upload your Excel file to begin processing with interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
