#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
MD_BIN=${MD_BIN:-"$HOME/.cargo/bin/md"}
RESULTS=${RESULTS:-"$ROOT/scaling-results.csv"}
TMP_ROOT=${TMP_ROOT:-/tmp/md-scaling-benchmark}
mkdir -p "$TMP_ROOT"
printf 'n,method,repeat,seconds,exit_status,sha256,command\n' > "$RESULTS"
hash=$(sha256sum "$MD_BIN" | awk '{print $1}')
for n in 100 400 1600; do
  for method in naive cells; do
    for repeat in 1 2 3; do
      out="$TMP_ROOT/n${n}-${method}-${repeat}"
      rm -rf "$out"
      command_text="$MD_BIN run --force $method --n $n --eq-steps 100 --steps 500 --out $out"
      seconds_file=$(mktemp)
      TIMEFORMAT='%3R'
      { time "$MD_BIN" run --force "$method" --n "$n" --eq-steps 100 --steps 500 --out "$out" >/dev/null 2>"$out.stderr"; } 2>"$seconds_file"
      status=$?
      seconds=$(tail -n 1 "$seconds_file")
      rm -f "$seconds_file" "$out.stderr"
      printf '%s,%s,%s,%s,%s,%s,"%s"\n' "$n" "$method" "$repeat" "$seconds" "$status" "$hash" "$command_text" >> "$RESULTS"
      if [ "$status" -ne 0 ]; then
        echo "FAILED: $command_text" >&2
        exit "$status"
      fi
      rm -rf "$out"
    done
  done
done
echo "Wrote $RESULTS"
