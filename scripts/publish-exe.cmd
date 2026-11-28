@echo off
rem Builds a single-file, self-contained amql-cli.exe for Windows x64.
rem Output: dist\win-x64\amql-cli.exe
rem
rem Not under bin\: that holds the framework-dependent builds, and a
rem self-contained one beside them stops their launchers (see
rem src\Directory.Build.props). Its intermediate build stays under obj\.

dotnet publish "%~dp0..\src\Amql.Cli" -c Release -p:PublishProfile=win-x64
if errorlevel 1 exit /b 1

echo.
echo Built: dist\win-x64\amql-cli.exe
