#!/usr/bin/env python3
"""Modular E2E and Unit Test Runner for CanvasIQ.

Supports executing individual tiers (Tier 1-4), specific milestones (M1-M5),
or specific features (1-28) using pytest (if installed) or standard library unittest.
"""

import os
import sys
import time
import json
import argparse
import unittest
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TIER_FILES = {
    1: "tests/e2e/test_tier1_feature_coverage.py",
    2: "tests/e2e/test_tier2_boundaries.py",
    3: "tests/e2e/test_tier3_interactions.py",
    4: "tests/e2e/test_tier4_real_world.py",
}

FEATURE_MILESTONE_MAP = {
    # M1
    1: "M1", 2: "M1", 3: "M1", 4: "M1", 5: "M1", 6: "M1", 7: "M1", 8: "M1",
    # M2
    9: "M2", 10: "M2", 11: "M2", 12: "M2", 13: "M2", 14: "M2", 15: "M2", 16: "M2", 17: "M2",
    # M3
    18: "M3", 19: "M3", 20: "M3", 21: "M3", 22: "M3", 23: "M3",
    # M4
    24: "M4", 25: "M4",
    # M1-M5
    26: "M5", 27: "M5", 28: "M5",
}


def run_with_pytest(test_files, pattern=None, verbose=False):
    """Executes tests using pytest if available."""
    try:
        import pytest
    except ImportError:
        return None

    args = ["-q"]
    if verbose:
        args = ["-v"]
    if pattern:
        args.extend(["-k", pattern])
    args.extend(test_files)
    ret = pytest.main(args)
    return ret == 0


def run_with_unittest(test_files, pattern=None, verbose=False):
    """Executes tests using standard library unittest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for file_path in test_files:
        module_path = file_path.replace(".py", "").replace("/", ".")
        try:
            mod_suite = loader.loadTestsFromName(module_path)
            if pattern:
                filtered_suite = unittest.TestSuite()
                for test_case in mod_suite:
                    for test in test_case:
                        if pattern.lower() in test.id().lower():
                            filtered_suite.addTest(test)
                suite.addTest(filtered_suite)
            else:
                suite.addTest(mod_suite)
        except Exception as e:
            print(f"[ERROR] Failed to load {module_path}: {e}", file=sys.stderr)
            return {"total": 0, "passed": 0, "skipped": 0, "failed": 1, "errors": [str(e)]}

    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.perf_counter()
    result = runner.run(suite)
    duration = time.perf_counter() - start_time

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)
    passed = total - failed - skipped

    return {
        "total": total,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "duration_sec": round(duration, 3),
        "success": result.wasSuccessful(),
    }


def main():
    parser = argparse.ArgumentParser(description="CanvasIQ Modular Test Runner")
    parser.add_argument("--tier", choices=["1", "2", "3", "4", "all"], default="all",
                        help="Run a specific test tier (1: Features, 2: Boundaries, 3: Interactions, 4: Real-World, all)")
    parser.add_argument("--milestone", choices=["M1", "M2", "M3", "M4", "M5", "all"], default="all",
                        help="Filter tests by milestone")
    parser.add_argument("--feature", type=int, choices=range(1, 29),
                        help="Run tests specifically targeting feature number (1-28)")
    parser.add_argument("--framework", choices=["auto", "pytest", "unittest"], default="auto",
                        help="Choose test execution framework (default: auto)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--json-report", "-j", type=str,
                        help="Output JSON summary report to specified path")

    args = parser.parse_args()

    # Determine files to run
    if args.tier == "all":
        selected_tiers = [1, 2, 3, 4]
    else:
        selected_tiers = [int(args.tier)]

    test_files = [TIER_FILES[t] for t in selected_tiers]

    pattern = None
    if args.feature:
        pattern = f"f{args.feature:02d}"
    elif args.milestone != "all":
        pattern = args.milestone

    print("=" * 70)
    print("CanvasIQ 4-Tier Opaque-Box E2E Test Runner")
    print(f"Selected Tiers: {selected_tiers} | Filter: {pattern or 'None'}")
    print("=" * 70)

    # Check framework
    use_pytest = False
    if args.framework in ("auto", "pytest"):
        try:
            import pytest
            use_pytest = True
        except ImportError:
            use_pytest = False

    tier_results = {}
    all_success = True

    for tier_num in selected_tiers:
        file_path = TIER_FILES[tier_num]
        print(f"\n---> Executing Tier {tier_num}: {file_path}")
        res = run_with_unittest([file_path], pattern=pattern, verbose=args.verbose)
        tier_results[f"tier_{tier_num}"] = res
        if not res["success"]:
            all_success = False

    # Summary
    print("\n" + "=" * 70)
    print("E2E Test Execution Summary:")
    print("-" * 70)
    total_tests = sum(r.get("total", 0) for r in tier_results.values())
    total_passed = sum(r.get("passed", 0) for r in tier_results.values())
    total_skipped = sum(r.get("skipped", 0) for r in tier_results.values())
    total_failed = sum(r.get("failed", 0) for r in tier_results.values())

    for tier_name, res in tier_results.items():
        status = "PASSED" if res.get("success", False) else "FAILED"
        print(f"  {tier_name.upper():<10}: {res.get('total', 0)} tests | {res.get('passed', 0)} passed | {res.get('skipped', 0)} skipped | {res.get('failed', 0)} failed ({res.get('duration_sec', 0)}s) [{status}]")

    print("-" * 70)
    print(f"  TOTAL     : {total_tests} tests | {total_passed} passed | {total_skipped} skipped | {total_failed} failed")
    print("=" * 70)

    if args.json_report:
        report_data = {
            "summary": {
                "total": total_tests,
                "passed": total_passed,
                "skipped": total_skipped,
                "failed": total_failed,
                "success": all_success,
            },
            "tiers": tier_results,
        }
        with open(args.json_report, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved JSON report to {args.json_report}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
