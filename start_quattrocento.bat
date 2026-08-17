@echo off
setlocal

echo Starting PatientGUI (Quattrocento / OTBioLab+)...
echo.
echo PREREQUISITE: OTBioLab+ must be running with its TCP/IP option enabled
echo and acquisition started. This server connects to OTBioLab+ on 127.0.0.1:31000,
echo decodes the stream, and serves it to the frontend just like the Ripple backend.
echo.

:: Set conda path - adjust if your Anaconda is installed elsewhere
set CONDA_PATH=%USERPROFILE%\\AppData\Local\miniconda3\Scripts\activate.bat

:: Path to your OTBioLab+ configuration file (extensionless XML / .otb+stp).
:: It sets the channel count and sample rate deterministically. EDIT THIS.
set OTB_CONFIG=%USERPROFILE%\Desktop\test1.otb+stp

:: Check if this is first-time setup
set NEEDS_SETUP=0

:: Check if conda environment exists
call "%CONDA_PATH%" base
conda env list | findstr /C:"patientgui" >nul 2>&1
if errorlevel 1 (
    echo [!] Conda environment 'patientgui' not found.
    set NEEDS_SETUP=1
)

:: Check if node_modules exists
if not exist "%~dp0fend\node_modules" (
    echo [!] Frontend dependencies not installed.
    set NEEDS_SETUP=1
)

:: If setup is needed, ask user
if %NEEDS_SETUP%==1 (
    echo.
    echo First-time setup required. Do you want to run setup now?
    choice /C YN /M "Run setup"
    if errorlevel 2 goto :skip_setup

    :: Create conda environment if needed
    conda env list | findstr /C:"patientgui" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Creating conda environment from patientgui.yml...
        conda env create -f "%~dp0bend\patientgui.yml"
        if errorlevel 1 (
            echo [ERROR] Failed to create conda environment.
            pause
            exit /b 1
        )
        echo Conda environment created successfully.
    )

    :: Install npm dependencies if needed
    if not exist "%~dp0fend\node_modules" (
        echo.
        echo Installing frontend dependencies...
        cd /d "%~dp0fend"
        npm install
        if errorlevel 1 (
            echo [ERROR] Failed to install npm dependencies.
            pause
            exit /b 1
        )
        echo Frontend dependencies installed successfully.
    )

    echo.
    echo Setup complete!
    echo.
)
:skip_setup

:: Start backend in a new window (with conda environment)
echo Starting Quattrocento backend server...
start "Backend Server (Quattrocento)" cmd /k "call %CONDA_PATH% patientgui && python -m server.quattrocento_server_cli --config ""%OTB_CONFIG%"" --port 8765 --pre-trigger 10 -o ""%~dp0output"""

:: Start frontend in a new window
echo Starting frontend dev server...
start "Frontend Dev Server" cmd /k "cd /d %~dp0fend && npm run dev"

echo.
echo Both servers are starting in separate windows.
endlocal
