# publish-linux.ps1 — builds AMQL for Linux from Windows: amql-gui (the text UI),
# amql-server and amql-cli, self-contained, in one folder to copy to the target
# (the DGX Spark by default). See publish-linux.sh for what the folder holds.
#
#   scripts\publish-linux.ps1                     # linux-arm64 → bin\linux-arm64
#   scripts\publish-linux.ps1 -Rid linux-x64      # x86-64 Linux, or WSL
#
# The CUDA library is not cross-compiled: build it on the target with
#   bash native/amql_cuda/build-cuda.sh
# and copy native/amql_cuda/bin/libamql_*.so beside amql-server.
param(
    [string]$Rid = "linux-arm64",
    [string]$Out = "",
    [string]$Configuration = "Release"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root "bin\$Rid" }
New-Item -ItemType Directory -Force $Out | Out-Null

foreach ($project in "Amql.Cli", "Amql.Server", "Amql.Gui") {
    Write-Host "[publish] $project -> $Out ($Rid)"
    dotnet publish (Join-Path $root "src\$project\$project.csproj") -c $Configuration -r $Rid --self-contained true `
        -p:PublishSingleFile=false -o $Out -nologo -v quiet
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# The Windows builds of the native libraries come along with Amql.Inference; they are no use on Linux.
Remove-Item (Join-Path $Out "amql_cuda.dll"), (Join-Path $Out "amql_inkling.dll") -ErrorAction SilentlyContinue

foreach ($lib in "libamql_cuda.so", "libamql_inkling.so") {
    $path = Join-Path $root "native\amql_cuda\bin\$lib"
    if (Test-Path $path) { Copy-Item $path $Out }
}
Write-Host "[publish] done: $Out\amql-gui  (copy the folder to the Linux machine; run ./amql-gui in a terminal)"
