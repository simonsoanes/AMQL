@echo off
rem Builds a single-file, self-contained amql-cli.exe for Windows x64.
rem Output: bin\Release\win-x64\publish\amql-cli.exe
rem
rem All projects under src\ share one output folder per configuration
rem (see src\Directory.Build.props), so this lands beside the ordinary
rem framework-dependent build rather than in a per-project bin.

dotnet publish "%~dp0..\src\Amql.Cli" -c Release -p:PublishProfile=win-x64
if errorlevel 1 exit /b 1

echo.
echo Built: bin\Release\win-x64\publish\amql-cli.exe
