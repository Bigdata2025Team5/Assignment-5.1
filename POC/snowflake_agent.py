import snowflake.connector
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_snowflake_data(year, quarter):
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role="ACCOUNTADMIN"
    )

    cursor = conn.cursor()
    cursor.execute("USE ROLE ACCOUNTADMIN;")
    cursor.execute("USE DATABASE ASSIGNMENT;")
    cursor.execute("USE SCHEMA NVDA_STAGE;")

    query = f"""
        SELECT metric, value
        FROM "{os.getenv("SNOWFLAKE_DATABASE")}"."{os.getenv("SNOWFLAKE_SCHEMA")}"."{os.getenv("SNOWFLAKE_TABLE")}"
        WHERE year = {year} AND quarter = {quarter}
    """
    cursor.execute(query)
    data = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(data, columns=columns)

    if df.empty:
        return {"summary": f"No financial data found for Q{quarter} {year}", "chart_data": []}

    df.columns = [col.lower() for col in df.columns]
    df["metric"] = df["metric"].str.strip().str.lower()

    def get_metric_value(name):
        row = df[df["metric"] == name.lower()]
        return row["value"].values[0] if not row.empty else "N/A"

    summary = (
        f"Valuation summary for Q{quarter} {year}:\n"
        f"• Market Cap: {get_metric_value('Market Cap')}\n"
        f"• Enterprise Value: {get_metric_value('Enterprise Value')}\n"
        f"• Trailing P/E: {get_metric_value('Trailing P/E')}\n"
        f"• Forward P/E: {get_metric_value('Forward P/E')}\n"
        f"• PEG Ratio (5yr expected): {get_metric_value('PEG Ratio (5yr expected)')}\n"
        f"• Price/Sales: {get_metric_value('Price/Sales')}\n"
        f"• Price/Book: {get_metric_value('Price/Book')}\n"
        f"• EV/Revenue: {get_metric_value('Enterprise Value/Revenue')}\n"
        f"• EV/EBITDA: {get_metric_value('Enterprise Value/EBITDA')}"
    )

    chart_metrics = ['trailing p/e', 'forward p/e', 'price/sales', 'price/book']
    df_chart = df[df["metric"].isin(chart_metrics)].copy()
    df_chart["metric"] = df_chart["metric"].str.title()
    df_chart["value_num"] = pd.to_numeric(df_chart["value"].str.replace("T", "e12", regex=False), errors="coerce")

    return {
        "summary": summary,
        "chart_data": df_chart[["metric", "value_num"]].to_dict(orient="records")
    }


