@echo off
rem build-inkling.cmd — build amql_inkling.dll, the Inkling 2 looped GPU path
rem (its own library: nothing in amql_cuda.dll changes). Same toolchain and
rem overrides as build-cuda.cmd: CUDA_ROOT, CUDA_ARCH (default sm_120), VCVARS,
rem VCVARS_VER (14.44: nvcc 13.3's cudafe++ crashes on 14.52's headers).

setlocal
cd /d "%~dp0"

if not defined CUDA_ROOT (
    for /d %%d in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*") do (
        set CUDA_ROOT=%%~fd
    )
)
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo [amql-inkling] nvcc not found at "%CUDA_ROOT%\bin\nvcc.exe"
    exit /b 1
)
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
    echo [amql-inkling] vcvars64.bat not found
    exit /b 1
)
if not defined CUDA_ARCH set CUDA_ARCH=sm_120
if not exist "bin" mkdir bin
if defined VCVARS_VER (
    call "%VCVARS%" -vcvars_ver=%VCVARS_VER% >nul 2>&1
) else (
    call "%VCVARS%" >nul 2>&1
)
"%CUDA_ROOT%\bin\nvcc.exe" -arch=%CUDA_ARCH% -O3 -std=c++17 -Xcompiler /MT --shared ^
    -o bin\amql_inkling.dll amql_inkling.cu -lcudart
if errorlevel 1 (
    echo [amql-inkling] nvcc build FAILED
    exit /b 1
)
echo [amql-inkling] built bin\amql_inkling.dll
exit /b 0
