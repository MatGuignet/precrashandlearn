"""
_harness.py — Minimal test harness for Crash&Learn test suite.
Shared by test_contract.py and test_integration.py.
"""
_pass = 0
_fail = 0

def ok(name: str) -> None:
    global _pass
    _pass += 1
    print(f"  PASS  {name}")

def fail(name: str, reason: str) -> None:
    global _fail
    _fail += 1
    print(f"  FAIL  {name}: {reason}")

def run_test(name: str, fn):
    import traceback
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, str(e))
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

def run_all(tests):
    import sys
    from env_simulation import close
    print(f"Running {len(tests)} tests...\n")
    for name, fn in tests:
        run_test(name, fn)
        close()
    print(f"\n{'='*50}")
    print(f"  {_pass} passed, {_fail} failed")
    sys.exit(1 if _fail > 0 else 0)