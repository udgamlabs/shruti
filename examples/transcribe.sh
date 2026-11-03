#!/usr/bin/env bash
# Transcribe a WAV file with Shruti using Cactus's needle command-line engine. No Python needed.
#
#   ./transcribe.sh clip.wav                          # language detected automatically
#   ./transcribe.sh talk.wav --audio-stream           # print the words as they are committed, chunk by chunk
#   ./transcribe.sh english.wav --audio-language en   # only for speech that is entirely English
#
# Any further options go straight to the engine (see `needle --help`: --audio-keywords, --audio-word-timestamps,
# --threads, ...). The WAV can be 8/16/24/32-bit PCM or float, any channel count and sample rate.
#
# The first run downloads two files and checks their SHA-256 before using them:
#   - the needle engine for this platform (about 1.5 MB) from Cactus-Compute/needle3 on Hugging Face, at a fixed
#     revision (Apache-2.0, by Cactus Compute)
#   - shruti.cact (29.5 MB) from udgamlabs/Shruti on Hugging Face
# Both are kept in $SHRUTI_CACHE (default ~/.cache/shruti). After that it runs offline; the engine makes no network calls.
# Output: one JSON object per line with "text" and "language" (plus "words" and "pending" with --audio-stream).
#
# Platforms: macOS on Apple silicon, Linux x86_64 and Linux arm64. On Windows, download windows-x86_64/needle.exe from
# the same revision and run: needle.exe --model shruti.cact --audio clip.wav
set -euo pipefail

NEEDLE_REPO="Cactus-Compute/needle3"
NEEDLE_REV="3391a1007e7db31d2fe5c69c734bbe93a9c1ce79"
# SHRUTI_MODEL=marathi ./transcribe.sh clip.wav picks Shruti Marathi; the default is Shruti Hindi (see SHA256SUMS).
case "${SHRUTI_MODEL:-hindi}" in
  hindi)   MODEL_URL="https://huggingface.co/udgamlabs/Shruti/resolve/main/shruti.cact"
           MODEL_SHA256="b189e954aaed4a9fd41acbad3ac05e486381e96325b22a4c695b9ea5153d7065" ;;
  marathi) MODEL_URL="https://huggingface.co/udgamlabs/shruti-marathi/resolve/main/shruti-marathi.cact"
           MODEL_SHA256="f51a92944cc25515e703518cfef5ec9c52bf0db4cf91741530ebdded1df5bb05" ;;
  tamil)   MODEL_URL="https://huggingface.co/udgamlabs/shruti-tamil/resolve/main/shruti-tamil.cact"
           MODEL_SHA256="8202f454350fb9ca2d2115ef5e81332c99de93b7c2f7c06fc9d2457995a4dec8" ;;
  bengali) MODEL_URL="https://huggingface.co/udgamlabs/shruti-bengali/resolve/main/shruti-bengali.cact"
           MODEL_SHA256="0a423c25f534e2099b43bc4e826018978c44a9f523cea19125686883a2a7c572" ;;
  telugu)  MODEL_URL="https://huggingface.co/udgamlabs/shruti-telugu/resolve/main/shruti-telugu.cact"
           MODEL_SHA256="197166720f32163bc2102ebf9dc0374d51ec9928175649071da92d943cca0d9e" ;;
  *) echo "transcribe.sh: SHRUTI_MODEL must be hindi, marathi, tamil, bengali or telugu" >&2; exit 1 ;;
esac
CACHE="${SHRUTI_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/shruti}"

die() {
  echo "transcribe.sh: $*" >&2
  exit 1
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
wav=$1
shift
[ -f "$wav" ] || die "no such file: $wav"

# The engine build for this machine, and its SHA-256 at revision $NEEDLE_REV.
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform=macos-arm64 needle_sha=346c108b7f3eaf8dffa43699ebef251d4d7c5f5257ccf1e225c1d11a15141537 ;;
  Linux-x86_64 | Linux-amd64) platform=linux-x86_64 needle_sha=388c2b91e1f36bfa29b7a3204310166483f6df7a388110eebf9885f6fcfa89fb ;;
  Linux-aarch64 | Linux-arm64) platform=linux-arm64 needle_sha=bc2b8cf8e27b92e8ad50ee091c0d461ddd2e27c560d461feca940457af591871 ;;
  *) die "this script has no engine build for $(uname -s) $(uname -m); see https://huggingface.co/$NEEDLE_REPO" ;;
esac

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# fetch URL DEST SHA256: download unless DEST already has the right hash; refuse a file with the wrong one.
fetch() {
  local url=$1 dest=$2 want=$3 got
  if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$want" ]; then return 0; fi
  echo "Downloading $url" >&2
  curl -fL --retry 3 --progress-bar -o "$dest.part" "$url" || die "download failed: $url"
  got=$(sha256 "$dest.part")
  if [ "$got" != "$want" ]; then
    rm -f "$dest.part"
    die "checksum mismatch for $url (got $got, expected $want)"
  fi
  mv "$dest.part" "$dest"
}

mkdir -p "$CACHE"
needle="$CACHE/needle-$platform-${NEEDLE_REV:0:7}"
fetch "https://huggingface.co/$NEEDLE_REPO/resolve/$NEEDLE_REV/$platform/needle" "$needle" "$needle_sha"
chmod +x "$needle"
model="$CACHE/${MODEL_URL##*/}"
fetch "$MODEL_URL" "$model" "$MODEL_SHA256"

exec "$needle" --model "$model" --audio "$wav" "$@"
