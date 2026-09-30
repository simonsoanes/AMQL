@echo off
rem build-cuda.cmd — build the optional amql_cuda native backend.
rem Requires: CUDA Toolkit (nvcc), MSVC x64 host toolchain (VS Build Tools).
rem Produces: native\amql_cuda\bin\amql_cuda.dll
rem Exit code 0 on success, 1 on any failure (missing toolchain, nvcc error).
rem
rem Environment overrides:
rem   CUDA_ROOT  — path to the CUDA Toolkit (default: auto-detect v13.x)
rem   CUDA_ARCH  — target SM architecture (default: sm_120; e.g. sm_121)
rem   VCVARS     — path to vcvars64.bat (default: VS 2022 Enterprise)
rem   CLI_OUT    — optional deployment directory (default: src\Amql.Cli\bin\Release\net10.0)

setlocal
cd /d "%~dp0"

rem ── CUDA Toolkit detection ───────────────────────────────────────────
if not defined CUDA_ROOT (
    for /d %%d in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*") do (
        set CUDA_ROOT=%%~fd
    )
)
if not defined CUDA_ROOT (
    set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
)
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo [amql-cuda] nvcc not found at "%CUDA_ROOT%\bin\nvcc.exe"
    echo [amql-cuda] set CUDA_ROOT to the CUDA Toolkit install and retry.
    exit /b 1
)

rem ── MSVC toolchain ───────────────────────────────────────────────────
if not defined VCVARS (
    set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat
)
if not exist "%VCVARS%" (
    for %%v in (Enterprise Professional Community BuildTools) do (
        if exist "C:\Program Files\Microsoft Visual Studio\18\%%v\VC\Auxiliary\Build\vcvars64.bat" (
            set VCVARS=C:\Program Files\Microsoft Visual Studio\18\%%v\VC\Auxiliary\Build\vcvars64.bat
        )
    )
)
if not exist "%VCVARS%" (
    echo [amql-cuda] vcvars64.bat not found — MSVC x64 toolchain required.
    echo [amql-cuda] set VCVARS to the full path of vcvars64.bat and retry.
    exit /b 1
)

rem ── Architecture ─────────────────────────────────────────────────────
if not defined CUDA_ARCH (
    set CUDA_ARCH=sm_120
)

rem ── Output ───────────────────────────────────────────────────────────
if not exist "bin" mkdir bin

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo [amql-cuda] vcvars64.bat failed.
    exit /b 1
)

echo [amql-cuda] nvcc:
"%CUDA_ROOT%\bin\nvcc.exe" --version | findstr /i release
echo [amql-cuda] compiling amql_cuda.cu -arch=%CUDA_ARCH% ...

"%CUDA_ROOT%\bin\nvcc.exe" -arch=%CUDA_ARCH% -O3 -std=c++17 ^
    -Xcompiler /MT ^
    --shared ^
    -o bin\amql_cuda.dll ^
    amql_cuda.cu ^
    -lcublasLt -lcublas -lcusolver -lcudart
if errorlevel 1 (
    echo [amql-cuda] nvcc build FAILED for -arch=%CUDA_ARCH%.
    exit /b 1
)

echo [amql-cuda] built bin\amql_cuda.dll

rem ── Stage next to the CLI ────────────────────────────────────────────
if not defined CLI_OUT (
    set CLI_OUT=D:\Dev\AMQL\src\Amql.Cli\bin\Release\net10.0
)
if exist "%CLI_OUT%" (
    copy /y bin\amql_cuda.dll "%CLI_OUT%\" >nul
    echo [amql-cuda] staged amql_cuda.dll into %CLI_OUT%
)
exit /b 0