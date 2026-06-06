"""
Backtest parity checker.

Verifies that every config constant imported by the live bot (orb_options_bot.py)
is also referenced somewhere in the NautilusTrader backtest files
(strategies/orb_options.py or backtest/run_orb_options_backtest.py).

Run this after every change to orb_options_bot.py or config.py:

    python check_backtest_parity.py

Exit 0 = in sync.  Exit 1 = drift detected — update strategies/orb_options.py.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_imports(path: Path) -> set[str]:
    """Return all names imported from 'config' in the given file."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "config":
            for alias in node.names:
                names.add(alias.name)
    return names


def _identifiers(path: Path) -> set[str]:
    """Return every identifier (Name node) referenced in a file."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


# ---------------------------------------------------------------------------
# Constants that are intentionally absent from the backtest
# ---------------------------------------------------------------------------

# These are live-account or screener concerns that have no equivalent in a
# pure bar-replay backtest and do not affect trade P&L calculations.
EXEMPT: set[str] = {
    # Account-level risk checks (no real account in backtest)
    "DAILY_LOSS_LIMIT_PCT",
    "MAX_DRAWDOWN_PCT",
    "MAX_RISK_PER_TRADE_PCT",
    "MIN_RR_RATIO",
    # Screener / universe selection (backtest symbols are passed on the CLI)
    "ORB_SCREENER_LIMIT",
    "ORB_SYMBOLS",
    # ORB equity-bot settings (options backtest only)
    "ORB_POSITION_SIZE",
    "MAX_TOTAL_INVESTMENT",
    # Multi-position budget cap — backtest runs each symbol in isolation
    "MAX_OPTIONS_INVESTMENT",
    # IC_MAX_DTE: backtest always runs intraday (always 0DTE by construction)
    "IC_MAX_DTE",
}

# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def main() -> int:
    live_imports   = _config_imports(ROOT / "orb_options_bot.py")
    strategy_ids   = _identifiers(ROOT / "strategies" / "orb_options.py")
    runner_ids     = _identifiers(ROOT / "backtest" / "run_orb_options_backtest.py")
    backtest_ids   = strategy_ids | runner_ids

    missing = live_imports - backtest_ids - EXEMPT

    if missing:
        print(f"\n  DRIFT DETECTED — {len(missing)} config item(s) used in live bot"
              " but missing from the backtest:\n")
        for name in sorted(missing):
            print(f"    ✗  {name}")
        print(
            "\n  Fix: add the missing constant(s) to strategies/orb_options.py\n"
            "  (import from config at module level and apply in _enter / on_bar).\n"
        )
        return 1

    covered = live_imports - EXEMPT
    print(f"\n  OK — all {len(covered)} live config imports are covered in the backtest"
          f" ({len(EXEMPT)} exempt).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
