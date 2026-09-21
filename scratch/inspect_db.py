import sqlite3
import pandas as pd

conn = sqlite3.connect('data/processed/trader.db')
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in c.fetchall()])

print("\n--- Event Log (Last 30) ---")
c.execute("SELECT id, level, component, message, details, timestamp FROM event_log ORDER BY id DESC LIMIT 30")
for r in c.fetchall():
    print(f"[{r[0]}] {r[5]} | {r[1]} | {r[2]} | {r[3]}")
    if r[4]:
        print(f"   Details: {r[4][:120]}...")

print("\n--- Predictions Table Summary ---")
c.execute("SELECT COUNT(*), MIN(date), MAX(date), COUNT(DISTINCT date) FROM predictions")
print("Total rows, Min date, Max date, Unique dates:", c.fetchone())

c.execute("SELECT model_version, COUNT(*) FROM predictions GROUP BY model_version")
print("Counts by model_version:", c.fetchall())

c.execute("SELECT date, COUNT(*) FROM predictions GROUP BY date ORDER BY date DESC LIMIT 15")
print("Predictions by date (recent 15):")
for r in c.fetchall():
    print(r)

print("\n--- Portfolio Snapshots ---")
c.execute("SELECT run_date, cash, total_value, positions_json FROM portfolio_snapshots ORDER BY run_date DESC LIMIT 5")
for r in c.fetchall():
    print(r[0], r[1], r[2], r[3][:100] if r[3] else None)
