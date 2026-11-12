#!/usr/bin/env bash
# Transcribe a WAV file with Shruti using Cactus's command-line engine. No Python needed.
#
#   ./transcribe.sh clip.wav                          # language detected automatically
#   ./transcribe.sh talk.wav --audio-stream           # print the words as they are committed, chunk by chunk
#   ./transcribe.sh english.wav --audio-language en   # only for speech that is entirely English
#
# Any further options go straight to the engine (see `cactus --help`: --audio-keywords, --audio-word-timestamps,
# --threads, ...). The WAV can be 8/16/24/32-bit PCM or float, any channel count and sample rate.
#
# The first run downloads two files and checks their SHA-256 before using them:
#   - the Cactus engine for this platform (about 1.2-1.6 MB) from Cactus-Compute/cactus on Hugging Face, at a fixed
#     revision, release 4.1.0 (Apache-2.0, by Cactus Compute)
#   - shruti.cact (29.5 MB) from udgamlabs/Shruti on Hugging Face
# Both are kept in $SHRUTI_CACHE (default ~/.cache/shruti). After that it runs offline; the engine makes no network calls.
# Output: one JSON object per line with "text" and "language" (plus "words" and "pending" with --audio-stream).
#
# Platforms: macOS on Apple silicon, Linux x86_64 and Linux arm64. On Windows, download releases/4.1.0/windows-x86_64/cactus.exe from
# the same revision and run: cactus.exe --model shruti.cact --audio clip.wav
set -euo pipefail

ENGINE_REPO="Cactus-Compute/cactus"
ENGINE_REV="4e822c9036be2eb0e47ef556df32a818bb1572ef"
ENGINE_RELEASE="releases/4.1.0"
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
  gujarati) MODEL_URL="https://huggingface.co/udgamlabs/shruti-gujarati/resolve/main/shruti-gujarati.cact"
           MODEL_SHA256="23fd52c8edb36b24383c70b82d6dd21bfa7b070ca1606b15510e75bbf3822e1f" ;;
  odia)    MODEL_URL="https://huggingface.co/udgamlabs/shruti-odia/resolve/main/shruti-odia.cact"
           MODEL_SHA256="e2818089abbd521b2224c01eec0d40e926673427d1abb0f78c27ebcfef71541d" ;;
  kannada) MODEL_URL="https://huggingface.co/udgamlabs/shruti-kannada/resolve/main/shruti-kannada.cact"
           MODEL_SHA256="492cdfdea934061edfefd14f2bd1e0255468adc8d0f0e3e9c2847cee69d5d5d8" ;;
  urdu)    MODEL_URL="https://huggingface.co/udgamlabs/shruti-urdu/resolve/main/shruti-urdu.cact"
           MODEL_SHA256="b5aaecd79cad932f24b1ab36ba61973d010bb8b14c1e8ee45ae9ad66955b4e8a" ;;
  malayalam) MODEL_URL="https://huggingface.co/udgamlabs/shruti-malayalam/resolve/main/shruti-malayalam.cact"
           MODEL_SHA256="5c8e3904c6fcbe984a26140835026ebb30f0df29596860ca5b9dabdc64513372" ;;
  assamese) MODEL_URL="https://huggingface.co/udgamlabs/shruti-assamese/resolve/main/shruti-assamese.cact"
           MODEL_SHA256="86b41d774be36b6eadc4fb0ae906a8f444705b822f5c433e68e49ff3de56cd0e" ;;
  sanskrit) MODEL_URL="https://huggingface.co/udgamlabs/shruti-sanskrit/resolve/main/shruti-sanskrit.cact"
           MODEL_SHA256="d6a0efbad1000edadad4334b1263bf860f8ef845c36ff2c81d83a9f8e8ad507c" ;;
  punjabi) MODEL_URL="https://huggingface.co/udgamlabs/shruti-punjabi/resolve/main/shruti-punjabi.cact"
           MODEL_SHA256="30d8f09ceac37a0d009f480896e3daac123dde94d1e2e913a0983ecdcde7fa36" ;;
  bodo) MODEL_URL="https://huggingface.co/udgamlabs/shruti-bodo/resolve/main/shruti-bodo.cact"
           MODEL_SHA256="fc38949cb343a7f914988f920286c6ee24d8166fafaad4ee4784af638b089580" ;;
  dogri) MODEL_URL="https://huggingface.co/udgamlabs/shruti-dogri/resolve/main/shruti-dogri.cact"
           MODEL_SHA256="5f0fc33adfc8bd1e2b0c973feaa34bf43b1c6f552ff2d0be300353d7339591bd" ;;
  kashmiri) MODEL_URL="https://huggingface.co/udgamlabs/shruti-kashmiri/resolve/main/shruti-kashmiri.cact"
           MODEL_SHA256="142cab90aba96992d3a0f5913ff72ba0e2097edb3aad613eeae247ff0cbc3504" ;;
  konkani) MODEL_URL="https://huggingface.co/udgamlabs/shruti-konkani/resolve/main/shruti-konkani.cact"
           MODEL_SHA256="c6c4de2cbc4c7e45af03cca684f4ed2c98c83ac5d68fa6044923a38b4ce902ff" ;;
  maithili) MODEL_URL="https://huggingface.co/udgamlabs/shruti-maithili/resolve/main/shruti-maithili.cact"
           MODEL_SHA256="7f87b9f2ce702956ad49bcec0f1da999bf4d38b3610b3d354ee9fac93f084842" ;;
  manipuri) MODEL_URL="https://huggingface.co/udgamlabs/shruti-manipuri/resolve/main/shruti-manipuri.cact"
           MODEL_SHA256="e3152c6b3ff1214ec9492811e51cd6e43037efdfb3d935df6238aca897975e5d" ;;
  nepali) MODEL_URL="https://huggingface.co/udgamlabs/shruti-nepali/resolve/main/shruti-nepali.cact"
           MODEL_SHA256="af3bf553a1448a468a376e491eedfaf5f53ee16ed9c6855173baa3d6c21aa7a0" ;;
  santali) MODEL_URL="https://huggingface.co/udgamlabs/shruti-santali/resolve/main/shruti-santali.cact"
           MODEL_SHA256="1faad55924ed6c01c3cff970c919b8c82c3ecc3712da7a22ce51f6344a125699" ;;
  sindhi) MODEL_URL="https://huggingface.co/udgamlabs/shruti-sindhi/resolve/main/shruti-sindhi.cact"
           MODEL_SHA256="d58f4646ca8fa87d62b0cea6ae53ba7c66fda6bd04ce18e2e37479ae72a79294" ;;
  *) echo "transcribe.sh: SHRUTI_MODEL must be a language in the README, in lowercase" >&2; exit 1 ;;
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

# The engine build for this machine, and its SHA-256 at revision $ENGINE_REV.
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) platform=macos-arm64 engine_sha=86c7bad785977c79d935a47604e14df0ef9640297b1b1280d93757a15a87f3f5 ;;
  Linux-x86_64 | Linux-amd64) platform=linux-x86_64 engine_sha=d3fabf298fed9da8db319c90fbce62c898930699d87c0468b20262860b337098 ;;
  Linux-aarch64 | Linux-arm64) platform=linux-arm64 engine_sha=088137b428c1be5cacb022a78a56f82815e5d60767002874afcc2ca7cf446074 ;;
  *) die "this script has no engine build for $(uname -s) $(uname -m); see https://huggingface.co/$ENGINE_REPO" ;;
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
engine="$CACHE/cactus-$platform-${ENGINE_REV:0:7}"
fetch "https://huggingface.co/$ENGINE_REPO/resolve/$ENGINE_REV/$ENGINE_RELEASE/$platform/cactus" "$engine" "$engine_sha"
chmod +x "$engine"
model="$CACHE/${MODEL_URL##*/}"
fetch "$MODEL_URL" "$model" "$MODEL_SHA256"

exec "$engine" --model "$model" --audio "$wav" "$@"
