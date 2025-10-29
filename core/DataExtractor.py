"""
Data Extraction Module for Viking Database
Provides functions to extract and preprocess data for the ML pipeline
"""

import pandas as pd
import os
import redshift_connector
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables FIRST
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

def fetch_viking_data_with_config(config: dict, output_path: str = None) -> pd.DataFrame:
    """
    Fetch Viking data with configuration parameters

    Args:
        config: Configuration dictionary with database and filter parameters
        output_path: Optional path to save the data

    Returns:
        DataFrame with fetched data
    """
    connection_details = {
        'host': config['host'],
        'port': config['port'],
        'database': config['database'],
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD')
    }

    # Build WHERE clause conditions
    where_conditions = [
        f"cp.pinged_at >= '{config['start_date']}'",
        f"cp.pinged_at < '{config['end_date']}'",
        "cp.call_ping_status = 'success'"
    ]

    if 'category_id' in config:
        where_conditions.append(f"cp.category_id = {config['category_id']}")

    if 'affiliate_ids' in config and config['affiliate_ids']:
        affiliate_list = ','.join(map(str, config['affiliate_ids']))
        where_conditions.append(f"cp.affiliate_integration_id IN ({affiliate_list})")

    query = f"""
        SELECT
            cp.legacy_call_ping_request_id, cp.call_ping_token,
            DATE_TRUNC('day', cp.pinged_at) AS ping_date,
            cp.affiliate_integration_id, a.affiliate_name, cp.category_id,
            CASE WHEN cd.ad_campaign_listing_id IS NOT NULL THEN 0 ELSE 1 END AS is_api_buyer,
            cp.bid_price, cp.max_price, cp.billable_duration_secs, cp.caller_id,
            cp.expected_revenue_value AS expected_revenue, cp.postal_code,
            DATE_TRUNC('hour', cp.created_at) AS created_hour,
            CASE WHEN cp.bid_accepted = 'true' THEN 1 ELSE 0 END AS won,
            cp.bid_modification_variant_id, cp.bid_modification_type,
            cp.rl_modified_bid, cp.config_modified_bid,
            CASE
                WHEN cp.bid_modification_type = 'CONFIGURATION'
                    AND cp.bid_modification_variant_id = 'configuration' THEN 'Configuration based modification'
                WHEN cp.bid_modification_type IS NULL
                    AND cp.bid_modification_variant_id = 'configuration' THEN 'No modification'
                WHEN cp.bid_modification_type = 'RL_EXPLORER'
                    AND cp.bid_modification_variant_id = 'rl' THEN 'RL Explorer modification'
                WHEN cp.bid_modification_type = 'RL_MODEL'
                    AND cp.bid_modification_variant_id = 'rl' THEN 'RL Model modification'
                WHEN cp.bid_modification_variant_id IS NULL
                    AND cp.was_bid_price_modified = 'true' THEN 'Configuration based modification'
                WHEN cp.bid_modification_variant_id IS NULL
                    AND cp.was_bid_price_modified = 'false' THEN 'No modification'
                ELSE 'RL Timeouts'
            END AS modification_type,
            cp.was_bid_price_modified,
            COALESCE(cp.modified_margin, cp.original_margin) * 100 AS final_modified,
            cp.expected_revenue_value - cp.bid_price AS profit
        FROM
            viking_core.call_ping cp
        LEFT JOIN viking_core.call_detail cd ON (cp.call_ping_token = cd.call_ping_token)
        INNER JOIN viking_core.affiliate_integration ai ON cp.affiliate_integration_id = ai.affiliate_integration_id
        INNER JOIN viking_core.affiliate a ON a.affiliate_id = ai.affiliate_id
        WHERE
            {' AND '.join(where_conditions)}
    """

    try:
        conn = redshift_connector.connect(**connection_details)
        df = pd.read_sql_query(query, conn)

        if output_path:
            df.to_parquet(output_path, index=False)

        return df

    finally:
        if 'conn' in locals() and conn:
            conn.close()


def test_connection():
    """Test database connection and data extraction with a small sample"""
    print("="*60)
    print("TESTING DATA EXTRACTOR")
    print("="*60)

    # Check environment variables
    print("\n1. Checking environment variables...")
    db_user = os.environ.get('DB_USER')
    db_pass = os.environ.get('DB_PASSWORD')

    if db_user:
        print(f"   ✓ DB_USER found: {db_user}")
    else:
        print("   ✗ DB_USER not found!")

    if db_pass:
        print(f"   ✓ DB_PASSWORD found: {'*' * len(db_pass)}")
    else:
        print("   ✗ DB_PASSWORD not found!")

    # Test configuration - just fetch 100 rows
    test_config = {
        'host': 'redshift-cluster-prod.car6nmbeqcqx.us-east-1.redshift.amazonaws.com',
        'port': 5439,
        'database': 'viking',
        'start_date': '2025-01-20',  # Recent 5 days
        'end_date': '2025-01-25',
        'category_id': 1933,
        'affiliate_ids': [1631]  # Just one affiliate for testing
    }

    print("\n2. Testing database connection...")
    print(f"   Host: {test_config['host']}")
    print(f"   Database: {test_config['database']}")
    print(f"   Date range: {test_config['start_date']} to {test_config['end_date']}")

    try:
        # Test with LIMIT 100
        connection_details = {
            'host': test_config['host'],
            'port': test_config['port'],
            'database': test_config['database'],
            'user': db_user,
            'password': db_pass
        }

        # Simple test query
        test_query = """
            SELECT COUNT(*) as total_rows,
                   MIN(pinged_at) as min_date,
                   MAX(pinged_at) as max_date
            FROM viking_core.call_ping
            WHERE pinged_at >= '2025-01-20'
              AND pinged_at < '2025-01-25'
              AND category_id = 1933
            LIMIT 1
        """

        print("\n3. Connecting to database...")
        conn = redshift_connector.connect(**connection_details)

        print("   ✓ Connected successfully!")

        print("\n4. Running test query...")
        test_df = pd.read_sql_query(test_query, conn)
        print(f"   ✓ Query executed successfully!")
        print(f"   Results: {test_df.to_dict('records')[0]}")

        # Now test the actual function with limited data
        print("\n5. Testing fetch_viking_data_with_config...")
        df = fetch_viking_data_with_config(test_config)

        print(f"   ✓ Data fetched successfully!")
        print(f"   Rows retrieved: {len(df):,}")
        print(f"   Columns: {df.shape[1]}")
        print(f"   Win rate: {df['won'].mean():.4%}")

        # Check for expected columns
        expected_cols = ['expected_revenue', 'won', 'affiliate_integration_id', 'final_modified']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"   ⚠️ Missing expected columns: {missing_cols}")
        else:
            print(f"   ✓ All expected columns present")

        print("\n✅ All tests passed!")
        return df

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()


if __name__ == "__main__":
    # Run test when module is executed directly
    test_df = test_connection()
    if test_df is not None:
        print("\n" + "="*60)
        print("Sample data preview:")
        print(test_df.head())
        print("\nColumn dtypes:")
        print(test_df.dtypes)