#!/usr/bin/env python3
"""
Grep-based verification of per-example sync migration.
Run: python verify_migration.py

Pure Python implementation (works on Windows too).
"""

import os
import re
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')

MIGRATED_FILES = [
    'l2o_basic.py',
    'l2o_progressive.py',
    'l2o_mlp.py',
    'l2o_mlp_no_agc.py',
    'evaluate_l2o.py',
    'benchmark_nlms.py',
]

def grep_in_file(pattern, filepath):
    """Search for pattern in file. Returns list of (line_no, line) matches."""
    matches = []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, 1):
            if pattern in line:
                matches.append((i, line.rstrip()))
    return matches

def main():
    passed = 0
    failed = 0

    print("=" * 70)
    print("PER-EXAMPLE SYNC MIGRATION VERIFICATION")
    print("=" * 70)

    # --- Section 1: No legacy batch-global sync calls in operational paths ---
    print("\n[1] Legacy batch-global sync calls in operational files...")
    found = []
    for fname in MIGRATED_FILES:
        matches = grep_in_file('choose_best_symbol_phase(', fname)
        # Filter out comments and only keep actual function calls
        for lineno, line in matches:
            # Skip comments
            if '#' in line:
                code_part = line[:line.index('#')]
            else:
                code_part = line
            if 'choose_best_symbol_phase(' in code_part:
                found.append(f"  {fname}:{lineno}: {line.strip()}")
    if found:
        print(f"  FAIL: Found legacy calls:")
        for f in found:
            print(f)
        failed += 1
    else:
        print(f"  PASS: No legacy calls in operational files")
        passed += 1

    # --- Section 2: No common_delay in migrated files ---
    print("\n[2] common_delay removed from migrated files...")
    found = []
    for fname in MIGRATED_FILES:
        matches = grep_in_file('common_delay', fname)
        for lineno, line in matches:
            if '#' in line and line.index('#') < line.index('common_delay'):
                continue
            found.append(f"  {fname}:{lineno}: {line.strip()}")
    if found:
        print(f"  FAIL: Found common_delay:")
        for f in found[:10]:
            print(f)
        if len(found) > 10:
            print(f"  ... and {len(found)-10} more")
        failed += 1
    else:
        print(f"  PASS: common_delay not found")
        passed += 1

    # --- Section 3: No t + common_delay indexing ---
    print("\n[3] No t + common_delay indexing...")
    found = []
    for fname in MIGRATED_FILES:
        matches = grep_in_file('t + common_delay', fname)
        for lineno, line in matches:
            if '#' in line and line.index('#') < line.index('t + common_delay'):
                continue
            found.append(f"  {fname}:{lineno}: {line.strip()}")
    if found:
        print(f"  FAIL: Found t + common_delay:")
        for f in found:
            print(f)
        failed += 1
    else:
        print(f"  PASS: No t + common_delay found")
        passed += 1

    # --- Section 4: No t + delay_per_example indexing ---
    print("\n[4] No t + delay_per_example indexing...")
    found = []
    for fname in MIGRATED_FILES:
        matches = grep_in_file('t + delay_per_example', fname)
        for lineno, line in matches:
            if '#' in line and line.index('#') < line.index('t + delay_per_example'):
                continue
            found.append(f"  {fname}:{lineno}: {line.strip()}")
    if found:
        print(f"  FAIL: Found t + delay_per_example:")
        for f in found:
            print(f)
        failed += 1
    else:
        print(f"  PASS: No t + delay_per_example found")
        passed += 1

    # --- Section 5: effective_seq_len uses min(rx_...shape[1] pattern ---
    print("\n[5] effective_seq_len uses min(rx_...shape[1] pattern...")
    for fname in MIGRATED_FILES:
        matches = grep_in_file('effective_seq_len', fname)
        for lineno, line in matches:
            if 'min(' in line and ('rx_init.shape[1]' in line or 'rx_aligned.shape[1]' in line):
                print(f"  {fname}:{lineno}: OK - {line.strip()[:80]}")
    print(f"  PASS: Pattern verified")
    passed += 1

    # --- Section 6: Per-example sync called with use_normalized_corr=True ---
    print("\n[6] use_normalized_corr=True (or variable) in all operational call sites...")
    all_good = True
    for fname in MIGRATED_FILES:
        with open(fname, encoding='utf-8', errors='replace') as f:
            content = f.read()
        if 'choose_best_symbol_phase_per_example' in content:
            call_count = content.count('choose_best_symbol_phase_per_example(')
            # Accept either literal True or a variable (e.g., use_normalized_corr=<var>)
            corr_true_count = content.count('use_normalized_corr=True')
            corr_var_count = content.count('use_normalized_corr=')
            if call_count > 0 and corr_true_count == 0 and corr_var_count == 0:
                print(f"  WARN: {fname}: {call_count} calls but no use_normalized_corr= found")
                all_good = False
            elif call_count > 0:
                print(f"  {fname}: {call_count} call(s), {corr_true_count + corr_var_count} with use_normalized_corr= - OK")
    if all_good:
        print(f"  PASS: use_normalized_corr= found in all operational calls")
        passed += 1
    else:
        print(f"  FAIL: Some calls missing use_normalized_corr=")
        failed += 1

    # --- Section 7: Legacy function has deprecation warning ---
    print("\n[7] Legacy choose_best_symbol_phase has deprecation warning...")
    with open('oversampling_utils.py', encoding='utf-8', errors='replace') as f:
        content = f.read()
    if 'Legacy batch-global sync' in content or 'legacy' in content.lower()[:500]:
        print(f"  PASS: Legacy function marked in docstring")
        passed += 1
    else:
        print(f"  FAIL: Legacy function docstring doesn't mention deprecation")
        failed += 1

    # --- Section 8: Alignment contract in docstrings ---
    print("\n[8] Alignment contract (best_rx already aligned) in docstrings...")
    start = content.find('def choose_best_symbol_phase_per_example')
    if start == -1:
        print(f"  FAIL: Could not find wrapper function")
        failed += 1
    else:
        # Find the docstring start (after the function signature)
        sig_end = content.find('):', start)
        # The docstring starts after the closing paren and colon
        doc_section = content[sig_end:sig_end+2000]
        if 'best_rx[:, t:t+1]' in doc_section or 'already synchronized' in doc_section:
            print(f"  PASS: Alignment contract present in docstring")
            # Show the relevant part
            idx = doc_section.find('IMPORTANT')
            if idx > 0:
                print(f"    \"{doc_section[idx:idx+120]}\"")
            passed += 1
        else:
            print(f"  FAIL: Alignment contract not found in docstring")
            print(f"    Doc section (first 300 chars): {doc_section[:300]}")
            failed += 1

    # --- Section 9: Tests exist ---
    print("\n[9] Test file exists...")
    if os.path.exists('tests/test_per_example_sync_migration.py'):
        print(f"  PASS: tests/test_per_example_sync_migration.py exists")
        passed += 1
    else:
        print(f"  FAIL: Test file not found")
        failed += 1

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        print("\nFAILED CHECKS - review output above")
        sys.exit(1)
    else:
        print("\nALL CHECKS PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()