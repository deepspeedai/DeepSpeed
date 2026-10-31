#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
#
# Run the tests/unit/v1 suite on the klxpu (KUNLUNXIN XPU) accelerator, one
# test file per subprocess.
#
# Per-file (not `pytest --forked`): torch_xmlir inits CUDA at import, so a fork
# per test hits "Cannot re-initialize CUDA in forked subprocess" for any test
# that touches CUDA. A fresh process per file avoids that.
#
# Hangs: DS_UNITTEST_TIMEOUT bounds a hung DistributedTest (framework calls
# pytest.exit, ending just this file); the outer `timeout -s KILL` backstops a
# non-distributed or collection-time hang.
#
# Blocklist (not allowlist) so new v1 tests run by default:
#   - ignore.txt    whole files/dirs to skip (--ignore)
#   - deselect.txt  individual node ids to skip (--deselect); the leading
#                   tests/ is stripped since pytest's rootdir is tests/.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || echo "$HERE/../../..")"
cd "$ROOT"

IGNORE_FILE="$HERE/ignore.txt"
DESELECT_FILE="$HERE/deselect.txt"

# Per-file wall-clock cap (outer) and per-test timeout (inner).
PER_FILE_KILL="${KLXPU_PER_FILE_KILL:-900}"
PER_TEST_TIMEOUT="${KLXPU_PER_TEST_TIMEOUT:-600}"
# Framework-side hang bound; keep under PER_FILE_KILL.
export DS_UNITTEST_TIMEOUT="${DS_UNITTEST_TIMEOUT:-600}"

readarray -t IGNORES < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$IGNORE_FILE")
# Deselect ids, normalized to rootdir-relative form (drop leading tests/).
readarray -t DESELECTS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$DESELECT_FILE" | sed -E 's#^tests/##')

is_ignored() {
    local f="$1" ig
    for ig in "${IGNORES[@]}"; do
        [[ "$f" == "$ig" || "$f" == "$ig"/* ]] && return 0
    done
    return 1
}

mapfile -t FILES < <(find tests/unit/v1 -name 'test_*.py' | sort)

npass=0
nfail=0
nskip=0
failed_files=()

for f in "${FILES[@]}"; do
    if is_ignored "$f"; then
        echo "IGNORE    $f"
        nskip=$((nskip + 1))
        continue
    fi
    # Deselect ids belonging to this file.
    frel="${f#tests/}"
    ds=()
    for d in "${DESELECTS[@]}"; do
        [[ "$d" == "$frel::"* ]] && ds+=("--deselect=$d")
    done
    timeout -s KILL "$PER_FILE_KILL" python -m pytest "$f" \
        -p no:randomly -q \
        --timeout="$PER_TEST_TIMEOUT" --timeout-method=signal \
        "${ds[@]}"
    rc=$?
    # 0 = passed, 5 = nothing collected (all skipped/deselected) -> both OK.
    if [[ $rc -eq 0 || $rc -eq 5 ]]; then
        echo "PASS      $f"
        npass=$((npass + 1))
    else
        echo "FAIL(rc=$rc) $f"
        nfail=$((nfail + 1))
        failed_files+=("$f")
    fi
done

echo "================= klxpu v1 summary ================="
echo "passed files : $npass"
echo "failed files : $nfail"
echo "ignored files: $nskip"
if ((nfail > 0)); then
    printf 'FAILED: %s\n' "${failed_files[@]}"
    exit 1
fi
