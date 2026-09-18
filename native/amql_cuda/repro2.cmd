@echo off
rem repro2.cmd — build & run the minimal raw-cublas isolation repro.
setlocal
cd /d "%~dp0"
set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set VCVARS=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul 2>&1
if errorlevel 1 ( echo vcvars FAILED & exit /b 1 )
"%CUDA_ROOT%\bin\nvcc.exe" -arch=sm_120 -O2 -o bin\repro2.exe repro2.cu -lcublasLt -lcublas -lcudart
if errorlevel 1 ( echo nvcc build FAILED & exit /b 1 )
bin\repro2.exe
exit /b %errorlevel%