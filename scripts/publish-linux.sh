#!/usr/bin/env bash
# publish-linux.sh — AMQL for a Linux machine (the DGX Spark by default), in one folder:
#   amql-gui     AMQL Studio's text UI (the same app as the Windows window)
#   amql-server  the inference server
#   amql-cli     every command the Studio's Tools page runs
#   libamql_cuda.so / libamql_inkling.so, when built (native/amql_cuda/build-cuda.sh)
#
# Self-contained: the folder carries its own .NET runtime, so the target needs
# no .NET install. Run it on the Spark itself, or on any machine with the .NET 11
# SDK (WSL, Windows via scripts/publish-linux.ps1) and copy the folder over.
#
#   scripts/publish-linux.sh                 # linux-arm64 → bin/linux-arm64
#   RID=linux-x64 scripts/publish-linux.sh   # an x86-64 Linux box (or WSL)
#   OUT=/opt/amql scripts/publish-linux.sh
#
# Then:  bin/linux-arm64/amql-gui       (in a terminal: ssh -t, tmux, a console)

set -euo pipefail
cd "$(dirname "$0")/.."

RID="${RID:-linux-arm64}"
OUT="${OUT:-bin/$RID}"
CONFIG="${CONFIG:-Release}"
mkdir -p "$OUT"

for project in Amql.Cli Amql.Server Amql.Gui; do
    echo "[publish] $project → $OUT ($RID)"
    dotnet publish "src/$project/$project.csproj" -c "$CONFIG" -r "$RID" --self-contained true \
        -p:PublishSingleFile=false -o "$OUT" -nologo -v quiet
done

# The Windows builds of the native libraries come along with Amql.Inference; they are no use here.
rm -f "$OUT/amql_cuda.dll" "$OUT/amql_inkling.dll"

staged=0
for lib in native/amql_cuda/bin/libamql_cuda.so native/amql_cuda/bin/libamql_inkling.so; do
    if [ -f "$lib" ]; then
        cp "$lib" "$OUT/"
        staged=1
    fi
done
if [ "$staged" = 0 ]; then
    echo "[publish] no CUDA library yet: on the target, run  bash native/amql_cuda/build-cuda.sh  and copy"
    echo "          native/amql_cuda/bin/libamql_*.so into $OUT (or set CLI_OUT=$OUT when building)."
fi

chmod +x "$OUT/amql-gui" "$OUT/amql-server" "$OUT/amql-cli" 2>/dev/null || true
echo "[publish] done: $OUT/amql-gui"
