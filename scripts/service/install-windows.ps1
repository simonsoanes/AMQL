<#
.SYNOPSIS
  Installs AMQL as a Windows service: amql-manager's supervisor, which runs the current
  version's manager and applies updates.

.DESCRIPTION
  Run in an elevated PowerShell. -Source is a release archive (amql-<version>-win-x64.zip)
  or a folder holding amql-manager.exe, amql-server.exe and amql-cli.exe (a publish, or
  bin/Release). The install goes to -Root:

    <Root>\versions\<version>\   this install
    <Root>\current               the version run
    <Root>\supervisor\           the service's own copy of amql-manager (never replaced by an update)
    <Root>\manager.json          the manager's config (written once; edit it to add models)
    <Root>\data\                 logs, job output, which models were running

  The service "AMQL" starts automatically and restarts on failure. The management API listens
  on -Urls; a key is generated unless one is given, and printed once.

.EXAMPLE
  scripts/service/install-windows.ps1 -Source dist/amql-1.1.0-win-x64.zip
  scripts/service/install-windows.ps1 -Source bin/Release -Urls http://0.0.0.0:8100
#>
param(
    [Parameter(Mandatory)] [string] $Source,
    [string] $Root = "$env:ProgramData\AMQL",
    [string] $Urls = 'http://127.0.0.1:8100',
    [string] $ApiKey,
    [string] $ServiceName = 'AMQL'
)
$ErrorActionPreference = 'Stop'
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'run this in an elevated PowerShell (it creates a service)'
}

$work = $null
if ((Get-Item $Source) -is [IO.FileInfo]) {
    $work = Join-Path $env:TEMP "amql-install-$([guid]::NewGuid().ToString('N'))"
    Expand-Archive $Source -DestinationPath $work
    $inner = Get-ChildItem $work
    $Source = if ($inner.Count -eq 1 -and $inner[0].PSIsContainer) { $inner[0].FullName } else { $work }
}
foreach ($exe in 'amql-manager.exe', 'amql-server.exe', 'amql-cli.exe') {
    if (-not (Test-Path (Join-Path $Source $exe))) { throw "$exe is not in $Source" }
}
$version = (& (Join-Path $Source 'amql-manager.exe') version).Trim()

New-Item -ItemType Directory -Force "$Root\versions", "$Root\data" | Out-Null
$target = "$Root\versions\$version"
if (Test-Path $target) { Remove-Item -Recurse -Force $target }
Copy-Item -Recurse $Source $target

if (Get-Service $ServiceName -ErrorAction SilentlyContinue) {
    Stop-Service $ServiceName -ErrorAction SilentlyContinue
}
if (Test-Path "$Root\supervisor") { Remove-Item -Recurse -Force "$Root\supervisor" }
Copy-Item -Recurse $Source "$Root\supervisor"

if (Test-Path "$Root\current") { Copy-Item "$Root\current" "$Root\previous" -Force }
Set-Content "$Root\current" $version -NoNewline

$printKey = $false
if (-not (Test-Path "$Root\manager.json")) {
    if (-not $ApiKey) {
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
        $ApiKey = 'amql-' + [Convert]::ToHexString($bytes).ToLowerInvariant()
        $printKey = $true
    }
    @{
        urls = $Urls
        api_key = $ApiKey
        bin_dir = $null
        containers_dir = "$Root\containers"
        models = @()
        update = @{ enabled = $true; auto_apply = $true; interval_hours = 24 }
    } | ConvertTo-Json -Depth 4 | Set-Content "$Root\manager.json"
    # The key is a secret: only administrators and the service read the config.
    icacls "$Root\manager.json" /inheritance:r /grant:r 'Administrators:F' 'SYSTEM:F' | Out-Null
}

$bin = "`"$Root\supervisor\amql-manager.exe`" supervise --root `"$Root`""
if (Get-Service $ServiceName -ErrorAction SilentlyContinue) {
    sc.exe config $ServiceName binPath= $bin start= auto | Out-Null
} else {
    New-Service -Name $ServiceName -BinaryPathName $bin -DisplayName 'AMQL' -StartupType Automatic `
        -Description 'AMQL inference and model management (amql-manager)' | Out-Null
}
sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/30000/restart/60000 | Out-Null
Start-Service $ServiceName
if ($work) { Remove-Item -Recurse -Force $work }

Write-Host "AMQL $version installed in $Root and running as the '$ServiceName' service."
Write-Host "Management API: $Urls (see docs/manager.md). Add models in $Root\manager.json, then Restart-Service $ServiceName."
if ($printKey) { Write-Host "API key (shown once, also in $Root\manager.json): $ApiKey" }
