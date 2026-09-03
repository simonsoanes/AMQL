@echo off
rem Builds a single-file, self-contained amql-cli.exe for Windows x64.
rem Output: src\Amql.Cli\bin\Release\net10.0\win-x64\publish\amql-cli.exe

dotnet publish "%~dp0..\src\Amql.Cli" -c Release -p:PublishProfile=win-x64
if errorlevel 1 exit /b 1

echo.
echo Built: src\Amql.Cli\bin\Release\net10.0\win-x64\publish\amql-cli.exe