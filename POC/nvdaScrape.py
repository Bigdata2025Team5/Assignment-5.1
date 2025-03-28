import os
import requests
import pandas as pd
import boto3
import snowflake.connector
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    # === Step 1: Scrape the valuation table from Yahoo Finance ===
    url = "https://finance.yahoo.com/quote/NVDA/key-statistics/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table")

    all_data = []
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                metric = cells[0].text.strip()
                value = cells[1].text.strip()
                all_data.append({"Metric": metric, "Value": value})

    df = pd.DataFrame(all_data)
    print("Scraped table:")
    print(df.head())

    # === Step 2: Upload to AWS S3 ===
    csv_path = "nvda_stats.csv"
    df.to_csv(csv_path, index=False)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    bucket_name = os.getenv("AWS_BUCKET_NAME")
    s3_key = "nvda/nvda_stats.csv"

    s3.upload_file(csv_path, bucket_name, s3_key)
    print(f"Uploaded to S3: s3://{bucket_name}/{s3_key}")

    # === Step 3: Upload to Snowflake stage ===
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role="ACCOUNTADMIN",
    )

    cursor = conn.cursor()

    stage_name = os.getenv("SNOWFLAKE_STAGE")
    storage_integration = os.getenv("SNOWFLAKE_STORAGE_INTEGRATION")

    cursor.execute(f"""
        CREATE OR REPLACE STAGE {stage_name}
        URL='s3://{bucket_name}/nvda/'
        STORAGE_INTEGRATION={storage_integration};
    """)
    print(f"Snowflake stage `{stage_name}` created.")

    cursor.execute(f"LIST @{stage_name};")
    print("Files in Snowflake Stage:")
    for row in cursor.fetchall():
        print(" -", row[0])

    # === Step 4: Create table and copy data from stage ===
    table_name = os.getenv("SNOWFLAKE_TABLE")

    cursor.execute(f"""
        CREATE OR REPLACE TABLE {table_name} (
            Year INT,
            Quarter INT,
            Metric STRING,
            Value STRING
        );
    """)

    cursor.execute(f"""
        COPY INTO {table_name}
        FROM @{stage_name}
        FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY='"' SKIP_HEADER=1);
    """)
    print(f"Data copied into Snowflake table `{table_name}`.")

    cursor.close()
    conn.close()

    # Optional: Delete the local CSV file after upload to S3
    os.remove(csv_path)

    print("Workflow finished successfully.")

except requests.exceptions.RequestException as e:
    print(f"Error during web scraping: {e}")
except snowflake.connector.errors.Error as e:
    print(f"Snowflake error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
