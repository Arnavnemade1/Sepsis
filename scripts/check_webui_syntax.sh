#!/bin/sh
# Parse every hand-written ES module in webui/.
#
# WHY THIS EXISTS: the twin's shaders are GLSL inside JS template literals, and a
# backtick in a GLSL comment silently TERMINATES the literal. The browser then
# reports "SyntaxError: missing ) after argument list" pointing at the .replace()
# call rather than at the comment, the whole module fails to load, and the only
# visible symptom is that a feature quietly does not appear. It has happened
# twice. A parse check costs nothing and catches it before a reload does.
#
# Vendored three.js is skipped: it is not ours to fix and it is large.
set -e
cd "$(dirname "$0")/.."
command -v node >/dev/null 2>&1 || { echo "node not found — skipping webui syntax check"; exit 0; }

tmp="${TMPDIR:-/tmp}/_webui_syntax.mjs"
fail=0
n=0
for f in webui/*.js; do
  case "$f" in *vendor*) continue;; esac
  n=$((n + 1))
  # No import stripping. `node --check` PARSES without resolving, so a bare
  # specifier like "three" is fine — and a first attempt that commented out
  # import lines with sed broke on camera.js, whose import spans several lines
  # and left a dangling `from "..."` that the checker then blamed.
  cp "$f" "$tmp"
  if node --check "$tmp" 2>/dev/null; then
    :
  else
    echo "FAIL  $f"
    node --check "$tmp" 2>&1 | head -4 | sed 's/^/      /'
    # The parser blames the .replace() call, which is nowhere near the cause.
    # Only run this hint on a file that ALREADY failed — a backtick in an
    # ordinary JS comment is perfectly legal, and guarding every file for it
    # flagged os.js and tactical-os.js, which have no shaders at all.
    if grep -n '^[[:space:]]*//.*`' "$f" >/dev/null 2>&1; then
      echo "      HINT: backtick inside a comment. If that comment is GLSL inside"
      echo "      a template literal, it ends the literal early:"
      grep -n '^[[:space:]]*//.*`' "$f" | head -3 | sed 's/^/        /'
    fi
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo "$n webui modules parse."
fi

exit "$fail"
