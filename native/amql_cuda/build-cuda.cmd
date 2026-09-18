@echo off
rem build-cuda.cmd — build the optional amql_cuda native backend.
rem Requires: CUDA Toolkit (nvcc), MSVC x64 host toolchain (VS Build Tools).
rem Produces: native\amql_cuda\bin\amql_cuda.dll
rem Exit code 0 on success, 1 on any failure (missing toolchain, nvcc error).

setlocal
cd /d "%~dp0"

set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo [amql-cuda] nvcc not found at "%CUDA_ROOT%\bin\nvcc.exe"
    echo [amql-cuda] set CUDA_ROOT to the CUDA Toolkit install and retry.
    exit /b 1
)

set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    echo [amql-cuda] vcvars64.bat not found — MSVC x64 toolchain required.
    exit /b 1
)

if not exist "bin" mkdir bin

call "%VCVARS%" >nul 2>&1
if errorlevel 1 (
    echo [amql-cuda] vcvars64.bat failed.
    exit /b 1
)

echo [amql-cuda] nvcc: 
"%CUDA_ROOT%\bin\nvcc.exe" --version | findstr /i release
echo [amql-cuda] compiling amql_cuda.cu -arch=sm_120 ...

"%CUDA_ROOT%\bin\nvcc.exe" -arch=sm_120 -O3 -std=c++17 ^
    -Xcompiler /MT ^
    --shared ^
    -o bin\amql_cuda.dll ^
    amql_cuda.cu ^
    -lcublasLt -lcublas -lcudart
if errorlevel 1 (
    echo [amql-cuda] nvcc build FAILED.
    exit /b 1
)

echo [amql-cuda] built bin\amql_cuda.dll

rem Stage the library next to the CLI so P/Invoke finds it at runtime.
set CLI_OUT=D:\Dev\AMQL\src\Amql.Cli\bin\Release\net10.0
if exist "%CLI_OUT%" (
    copy /y bin\amql_cuda.dll "%CLI_OUT%\" >nul
    echo [amql-cuda] staged amql_cuda.dll into %CLI_OUT%
)
exit /b 0