import time
from app.risk_tools import calculate_tail_risk_metrics, analyze_debt_maturity_wall
from app.quant_tools import calculate_factor_information_coefficient, validate_signal_statistics
from app.macro_tools import fetch_india_macro_data

def run_tests():
    ticker = "TATAPOWER.NS"

    print(f"--- Testing {ticker} ---")
    
    print("\n1. calculate_tail_risk_metrics")
    res = calculate_tail_risk_metrics(ticker)
    print(res)

    print("\n2. analyze_debt_maturity_wall")
    res = analyze_debt_maturity_wall(ticker)
    print(res)

    print("\n3. calculate_factor_information_coefficient")
    res = calculate_factor_information_coefficient(ticker)
    print(res)

    print("\n4. validate_signal_statistics")
    res = validate_signal_statistics(ticker)
    print(res)

    print("\n5. fetch_india_macro_data (testing pandas read_json fix)")
    res = fetch_india_macro_data()
    print(res)

if __name__ == "__main__":
    run_tests()
