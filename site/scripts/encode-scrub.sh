#!/usr/bin/env bash
# Re-encode a clip for SCROLL SCRUBBING.
#
# The one thing that matters here is keyframe density. A normally-encoded clip
# keeps very few keyframes — the source for the technology stack had exactly TWO
# across six seconds — and every backward seek then has to decode forward from
# the previous one. That is what makes scrolling up through a scrubbed video
# stutter or sit frozen.
#
# `-g 1` makes every frame independently decodable, so a seek to any position is
# a single decode. On this footage that cost almost nothing: 2.3 MB -> 2.8 MB,
# because the clip is small, dark and smooth, so all-intra compresses well. Do
# not assume that holds for busier material — check the output size.
#
# Usage: ./scripts/encode-scrub.sh in.mp4 public/media/out.mp4 [crf]
set -euo pipefail

IN="${1:?input file}"
OUT="${2:?output file}"
CRF="${3:-20}"

ffmpeg -y -i "$IN" -an \
  -c:v libx264 -preset slow -profile:v high -pix_fmt yuv420p \
  -x264-params "keyint=1:min-keyint=1:scenecut=0" \
  -crf "$CRF" -movflags +faststart "$OUT"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "keyframes: $(ffprobe -v error -select_streams v:0 -skip_frame nokey \
  -show_entries frame=pts_time -of csv=p=0 "$OUT" | wc -l | tr -d ' ')"
