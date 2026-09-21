import sys
import io
from streamlit.testing.v1 import AppTest

# Set utf-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=== RUNNING STREAMLIT APPTEST AUDIT ===")
at = AppTest.from_file("dashboard/app.py", default_timeout=30)
at.run()

if at.exception:
    print(f"FAILED: App raised exceptions: {at.exception}")
    sys.exit(1)

print("SUCCESS: dashboard/app.py rendered cleanly with 0 exceptions!\n")

print("--- 1. SUBHEADERS FOUND IN RENDERED STREAMLIT APP ---")
for s in at.subheader:
    print(f"  • {s.value}")

print("\n--- 2. METRICS RENDERED IN TAB 1 (PORTFOLIO INTELLIGENCE) ---")
for m in at.metric:
    label = m.label
    val = m.value
    help_text = getattr(m, "help", "")
    if any(k in label.lower() for k in ["sector", "diversification", "cash reserve", "exposure", "active"]):
        print(f"  • {label:<25} = {val}")

print("\n--- 3. METRICS RENDERED IN TAB 3 (MODEL DRIFT & PSI MONITOR) ---")
for m in at.metric:
    label = m.label
    val = m.value
    if any(k in label.lower() for k in ["psi", "ks test", "live vs base", "drift"]):
        print(f"  • {label:<25} = {val}")

print("\n--- 4. NOTIFICATIONS, DIAGNOSES & DISCLAIMERS ---")
for info in at.info:
    print(f"  [Info] {info.value}")

for cap in at.caption:
    if any(k in cap.value.lower() for k in ["drift", "formula", "breadth", "disclaimer", "sector"]):
        print(f"  [Caption] {cap.value}")
