#!/usr/bin/env python3
"""
Check the test database structure and content.
"""

import sqlite3
from pathlib import Path

import pandas as pd

db_path = Path("tests/resources/wikipedia/test_wikipedia.db")
conn = sqlite3.connect(str(db_path))

# Check what tables exist
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

# Check wiki_views table
df_wiki = pd.read_sql("SELECT * FROM wiki_views LIMIT 5", conn)
print("\nWiki views sample:")
print(df_wiki.head())

# Check stock_prices table
df_prices = pd.read_sql("SELECT * FROM stock_prices LIMIT 5", conn)
print("\nStock prices sample:")
print(df_prices.head())

# Check date ranges
wiki_dates = pd.read_sql("SELECT MIN(date), MAX(date), COUNT(*) FROM wiki_views", conn)
price_dates = pd.read_sql("SELECT MIN(date), MAX(date), COUNT(*) FROM stock_prices", conn)
print("\nWiki dates:", wiki_dates.values[0])
print("Price dates:", price_dates.values[0])

# Check DAX stocks specifically
dax_stocks = ["ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BMW.DE"]
print(f"\nChecking DAX stocks: {dax_stocks}")

for stock in dax_stocks:
    if stock in df_wiki.columns:
        wiki_count = df_wiki[stock].notna().sum()
        print(f"{stock} wiki views: {wiki_count} non-null values")
    else:
        print(f"{stock} not found in wiki_views")

    if stock in df_prices.columns:
        price_count = df_prices[stock].notna().sum()
        print(f"{stock} stock prices: {price_count} non-null values")
    else:
        print(f"{stock} not found in stock_prices")

conn.close()
