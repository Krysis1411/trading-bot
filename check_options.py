"""
check_options.py — Verify options access on Alpaca paper account
and fetch a live options chain for NVDA as a test.
"""
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import ContractType
from datetime import date, timedelta

load_dotenv()

key    = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY")
secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")

client = TradingClient(key, secret, paper=True)

# --- Account info ---
account = client.get_account()
print("=" * 50)
print(f"Account:              {account.id}")
print(f"Buying Power:         ${float(account.buying_power):,.2f}")
print(f"Options Level:        {account.options_trading_level}")
print(f"Options Approved:     {account.options_approved_level}")
print("=" * 50)

# --- Fetch a live NVDA options chain (nearest expiry) ---
expiry = date.today() + timedelta(days=14)  # next 2 weeks

contracts = client.get_option_contracts(GetOptionContractsRequest(
    underlying_symbols=["NVDA"],
    type=ContractType.CALL,
    expiration_date_lte=str(expiry),
    expiration_date_gte=str(date.today()),
    limit=5,
))

print("\nSample NVDA CALL contracts available:")
print(f"{'Symbol':<30} {'Strike':>8} {'Expiry':<12}")
print("-" * 55)
for c in contracts.option_contracts:
    print(f"{c.symbol:<30} ${float(c.strike_price):>7.2f}  {str(c.expiration_date):<12}")

print("\n✅ Options access confirmed. Ready to trade!")
