@echo off
setlocal

echo Starting PatientGUI (Quattrocento / OTBioLab+)...
echo.
echo PREREQUISITE: OTBioLab+ must be running with its TCP/IP option enabled
echo and acquisition started. This server connects to OTBioLab+ on 127.0.0.1:31000,
echo decodes the stream, and serves it to the frontend just like the Ripple backend.
echo.

:: ---------------------------------------------------------------------------
:: If auto-detection below cannot find your conda installation, put the full
:: path to its Scripts\activate.bat here and this script will use it verbatim.
:: ---------------------------------------------------------------------------
set "CONDA_OVERRIDE="

:: Path to your OTBioLab+ configuration file (extensionless XML / .otb+stp).
:: It sets the channel count and sample rate deterministically. EDIT THIS.
set "OTB_CONFIG=%USERPROFILE%\Desktop\test1.otb+stp"


call :find_conda
if errorlevel 1 exit /b 1

:: --- first-time setup -------------------------------------------------------
set NEEDS_SETUP=0

call "%CONDA_PATH%" base
call conda env list | findstr /C:"patientgui" >nul 2>&1
if errorlevel 1 (
    echo [!] Conda environment 'patientgui' not found.
    set NEEDS_SETUP=1
)

if not exist "%~dp0fend\node_modules" (
    echo [!] Frontend dependencies not installed.
    set NEEDS_SETUP=1
)

if %NEEDS_SETUP%==1 (
    echo.
    echo First-time setup required. Do you want to run setup now?
    choice /C YN /M "Run setup"
    if errorlevel 2 (
        echo.
        echo [ABORT] Setup declined. The platform cannot start without it -
        echo         the backend would fail immediately with "No module named server".
        pause
        exit /b 1
    )
    call :run_setup
    if errorlevel 1 exit /b 1
)

:: --- refuse to launch someone else's checkout -------------------------------
call :check_provenance
if errorlevel 1 exit /b 1


:: --- refuse to launch quietly behind a stale backend ------------------------
call :check_port
if errorlevel 1 exit /b 1

:: --- launch -----------------------------------------------------------------
:: Build each command line in a variable first. Nesting doubled quotes directly
:: inside `start "title" cmd /k "..."` breaks the moment either the conda path
:: or the repository path contains a space - verified in cmd, not theoretical.
set "BACKEND_CMD=call "%CONDA_PATH%" patientgui && python -m server.quattrocento_server_cli --config "%OTB_CONFIG%" --port 8765 --pre-trigger 10 -o "%~dp0output""
echo.
echo Starting backend server...
start "Backend Server (Quattrocento)" cmd /k "%BACKEND_CMD%"

set "FRONTEND_CMD=cd /d "%~dp0fend" && npm run dev"
echo Starting frontend dev server...
start "Frontend Dev Server" cmd /k "%FRONTEND_CMD%"

echo.
echo Both servers are starting in separate windows.
endlocal
exit /b 0


:: ===========================================================================
:: Subroutines
:: ===========================================================================

:find_conda
:: Probe the usual installation locations instead of hardcoding one. The three
:: launchers used to disagree about where conda lives, so on any given machine
:: at least one of them was dead.
set "CONDA_PATH="
if defined CONDA_OVERRIDE if exist "%CONDA_OVERRIDE%" set "CONDA_PATH=%CONDA_OVERRIDE%"
for %%D in (
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%PROGRAMDATA%\miniconda3"
) do (
    if not defined CONDA_PATH if exist "%%~D\Scripts\activate.bat" set "CONDA_PATH=%%~D\Scripts\activate.bat"
)
if not defined CONDA_PATH (
    echo [ERROR] Could not find a conda installation. Looked for
    echo         Scripts\activate.bat under:
    echo           %USERPROFILE%\miniconda3
    echo           %LOCALAPPDATA%\miniconda3
    echo           %USERPROFILE%\anaconda3
    echo           %LOCALAPPDATA%\anaconda3
    echo           %PROGRAMDATA%\miniconda3
    echo.
    echo         If conda is installed somewhere else, set CONDA_OVERRIDE near
    echo         the top of this script to the full path of its activate.bat.
    pause
    exit /b 1
)
echo Using conda: %CONDA_PATH%
exit /b 0


:run_setup
:: Create the environment if it is missing.
call conda env list | findstr /C:"patientgui" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Creating conda environment from patientgui.yml...
    call conda env create -f "%~dp0bend\patientgui.yml"
    if errorlevel 1 (
        echo [ERROR] Failed to create conda environment.
        pause
        exit /b 1
    )
    echo Conda environment created successfully.
)

:: Install the backend package. This step was documented in bend/pyproject.toml
:: and performed by nothing, so a clean machine got "No module named server"
:: from a window that otherwise looked like a healthy server. Runs whether the
:: environment was just created or already existed.
echo.
echo Installing the backend package in editable mode...
call "%CONDA_PATH%" patientgui
call pip install -e "%~dp0bend" --no-deps
if errorlevel 1 (
    echo [ERROR] Failed to install the backend package from
    echo         %~dp0bend
    pause
    exit /b 1
)
echo Backend package installed.

:: Install frontend dependencies. pushd/popd, because the old `cd /d` leaked
:: into the launch step's working directory.
if not exist "%~dp0fend\node_modules" (
    echo.
    echo Installing frontend dependencies...
    pushd "%~dp0fend"
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] Failed to install npm dependencies.
        pause
        exit /b 1
    )
    popd
    echo Frontend dependencies installed successfully.
)

echo.
echo Setup complete!
echo.
exit /b 0


:check_provenance
:: `server` is imported by name, so whichever editable install the environment
:: happens to carry wins - including one left behind by a different clone of
:: this repository. Launching that would run code nobody is looking at.
echo.
echo Checking that the backend package points at this folder...
call "%CONDA_PATH%" patientgui >nul 2>&1
python -c "import importlib.util as u,os,sys; s=u.find_spec('server'); p=(list(s.submodule_search_locations) or [None])[0] if s else None; g=os.path.normcase(os.path.abspath(os.path.dirname(p))) if p else ''; print('    backend package -> ' + (g or '(unresolved)')); sys.exit(0 if g==os.path.normcase(os.path.abspath(sys.argv[1])) else 1)" "%~dp0bend\src" 2>nul
if not errorlevel 1 goto :provenance_ok
echo.
echo [!] The 'server' package does NOT resolve to this checkout:
echo         this folder    %~dp0bend\src
echo     The backend would run a DIFFERENT copy of the code than the one in
echo     this folder, silently. This is usually a leftover editable install
echo     from another clone.
echo.
choice /C YN /M "Repair now by reinstalling the backend package from this folder"
if errorlevel 2 (
    echo.
    echo [ABORT] Not launching against a different checkout.
    pause
    exit /b 1
)
call pip install -e "%~dp0bend" --no-deps
if errorlevel 1 (
    echo [ERROR] Repair failed.
    pause
    exit /b 1
)
python -c "import importlib.util as u,os,sys; s=u.find_spec('server'); p=(list(s.submodule_search_locations) or [None])[0] if s else None; g=os.path.normcase(os.path.abspath(os.path.dirname(p))) if p else ''; print('    backend package -> ' + (g or '(unresolved)')); sys.exit(0 if g==os.path.normcase(os.path.abspath(sys.argv[1])) else 1)" "%~dp0bend\src" 2>nul
if errorlevel 1 (
    echo [ERROR] Still resolving elsewhere after the repair. Check for a
    echo         conflicting .pth or an installed 'server' package in the
    echo         patientgui environment.
    pause
    exit /b 1
)
:provenance_ok
echo     OK - backend package resolves to this checkout.
exit /b 0


:check_port
:: A backend from an earlier run keeps port 8765. The new one then dies on
:: bind inside a window that stays open, while the frontend connects to the
:: OLD instance - different config, possibly a half-open recording.
netstat -ano -p TCP 2>nul | findstr /C:":8765" | findstr /C:"LISTENING" >nul
if errorlevel 1 exit /b 0
echo.
echo [!] Port 8765 is already in use - a backend is probably still running.
echo     If you continue, the new backend will FAIL to bind and the frontend
echo     will silently connect to the old one instead.
echo.
echo     Close the previous "Backend Server" window first. To find the owner:
echo         netstat -ano ^| findstr :8765
echo.
choice /C YN /M "Continue anyway"
if errorlevel 2 (
    echo.
    echo [ABORT] Close the stale backend and run this again.
    pause
    exit /b 1
)
exit /b 0
