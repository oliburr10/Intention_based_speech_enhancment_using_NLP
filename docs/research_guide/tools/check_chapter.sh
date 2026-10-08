#!/usr/bin/env bash
# Compile the guide with only ONE chapter/appendix included, in an isolated
# build directory, so several people (or agents) can check their own file
# concurrently without clobbering each other's aux files.
#
#   tools/check_chapter.sh chapters/05-codesign
#   tools/check_chapter.sh appendices/b-registers
#
# Cross-references to other chapters will show as ?? -- that is expected.
# Exit status is non-zero if pdflatex reports an error.
set -euo pipefail
cd "$(dirname "$0")/.."
part="${1:?usage: tools/check_chapter.sh chapters/NN-name}"
part="${part%.tex}"
name="$(basename "$part")"
out="${TMPDIR:-/tmp}/rg-check-${name}"
mkdir -p "$out"/{frontmatter,chapters,appendices}
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir="$out" -jobname="$name" \
  -pdflatex="pdflatex %O '\\includeonly{${part}}\\input{%S}'" \
  main.tex >"$out/latexmk.log" 2>&1 || {
    echo "BUILD FAILED for $part -- first errors:"; grep -nE '^(\./)?[^ ]+\.tex:[0-9]+:|^! ' "$out/$name.log" | head -20 || tail -40 "$out/latexmk.log"; exit 1; }
echo "OK: $out/$name.pdf"
echo "Warnings in $part (undefined refs to other chapters are expected):"
grep -E 'Overfull|LaTeX Warning: (Citation|Reference).*undefined|Package biblatex Warning|Missing character' "$out/$name.log" | sort | uniq -c | sort -rn | head -25 || true
