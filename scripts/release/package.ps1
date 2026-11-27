<#
.SYNOPSIS
  Builds one AMQL release archive per platform, in the form amql-manager's updater installs.

.DESCRIPTION
  For each runtime: a self-contained publish of amql-manager, amql-server, amql-cli and
  amql-gui (the WPF window on Windows, the text UI on Linux), with the native CUDA libraries
  built for that platform, archived as dist/amql-<version>-<rid>.zip (Windows) or .tar.gz
  (Linux) with everything inside one amql/ folder. dist/SHA256SUMS lists them all.

  Publish a release by attaching every file in dist/ to a GitHub release tagged v<version>:
    gh release create v<version> dist/* --title "AMQL <version>" --notes-file CHANGELOG-excerpt.md
  An install's manager finds it on its daily check (or POST /v1/manage/update/check), checks
  the archive against SHA256SUMS and installs it.

.EXAMPLE
  scripts/release/package.ps1 -Version 1.1.0
  scripts/release/package.ps1 -Version 1.1.0 -Runtimes win-x64,linux-arm64
#>
param(
    [Parameter(Mandatory)] [string] $Version,
    [string[]] $Runtimes = @('win-x64', 'linux-x64', 'linux-arm64'),
    [string] $Out = 'dist'
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '../..')
if ($Version -notmatch '^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$') { throw "version '$Version' is not major.minor.patch[-label]" }
New-Item -ItemType Directory -Force $Out | Out-Null
$sums = Join-Path $Out 'SHA256SUMS'

foreach ($rid in $Runtimes) {
    $stage = Join-Path $Out "stage-$rid/amql"
    if (Test-Path (Split-Path $stage)) { Remove-Item -Recurse -Force (Split-Path $stage) }
    New-Item -ItemType Directory -Force $stage | Out-Null
    foreach ($project in 'Amql.Manager', 'Amql.Server', 'Amql.Cli', 'Amql.Gui') {
        Write-Host "[package] $project -> $rid"
        dotnet publish "src/$project/$project.csproj" -c Release -r $rid --self-contained true `
            -p:Version=$Version -p:PublishSingleFile=false -o $stage -nologo -v quiet
        if ($LASTEXITCODE -ne 0) { throw "publish of $project for $rid failed" }
    }
    # Native libraries: the platform's own builds only.
    if ($rid.StartsWith('win')) {
        Get-ChildItem $stage -Filter 'libamql_*.so' | Remove-Item
    } else {
        Get-ChildItem $stage -Filter 'amql_*.dll' | Remove-Item
        foreach ($lib in 'native/amql_cuda/bin/libamql_cuda.so', 'native/amql_cuda/bin/libamql_inkling.so') {
            if (Test-Path $lib) { Copy-Item $lib $stage }
            else { Write-Warning "$lib is not built: the $rid archive runs on the CPU only (native/amql_cuda/build-cuda.sh)" }
        }
    }
    foreach ($exe in 'amql-manager', 'amql-server', 'amql-cli') {
        $name = if ($rid.StartsWith('win')) { "$exe.exe" } else { $exe }
        if (-not (Test-Path (Join-Path $stage $name))) { throw "$name is missing from the $rid publish" }
    }

    $base = "amql-$Version-$rid"
    if ($rid.StartsWith('win')) {
        $archive = Join-Path $Out "$base.zip"
        if (Test-Path $archive) { Remove-Item $archive }
        Compress-Archive -Path $stage -DestinationPath $archive
    } else {
        $archive = Join-Path $Out "$base.tar.gz"
        if (Test-Path $archive) { Remove-Item $archive }
        # tar keeps the executables' mode bits only when made where they exist; Windows' tar
        # writes 0644, so the updater sets the execute bit on the AMQL executables it installs.
        tar -czf $archive -C (Split-Path $stage) amql
        if ($LASTEXITCODE -ne 0) { throw "tar failed for $rid" }
    }
    Remove-Item -Recurse -Force (Split-Path $stage)
    Write-Host "[package] $archive"
}

$lines = Get-ChildItem $Out -File | Where-Object { $_.Name -like "amql-$Version-*" } | Sort-Object Name | ForEach-Object {
    "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $_.Name
}
Set-Content -Path $sums -Value $lines -Encoding ascii
Write-Host "[package] $sums"
Get-Content $sums
