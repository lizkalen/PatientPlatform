@echo off
:: Delayed expansion is on for the whole script, and every path-bearing value
:: below is referenced as !VAR! rather than %VAR%. Reason: cmd expands %VAR% in
:: the parsing pass, BEFORE it looks for command separators, so a literal & in
:: a repository or conda path silently truncates the line it appears in - the
:: backend command loses everything after `-o `, the frontend command collapses
:: to a bare `cd /d`, and the diagnostic echoes turn into "not recognized"
:: errors. !VAR! is substituted after that pass and is immune.
:: The tradeoff: delayed expansion eats a literal ! in a path. A & in a folder
:: name is common ("Lab & Co"); a ! is not. Chosen deliberately.
setlocal enabledelayedexpansion

echo Starting PatientGUI...
echo.

:: ---------------------------------------------------------------------------
:: If auto-detection below cannot find your conda installation, put the full
:: path to its Scripts\activate.bat here and this script will use it verbatim.
:: ---------------------------------------------------------------------------
set "CONDA_OVERRIDE="

:: This script's own folder, captured once. Assigning it inside a quoted set is
:: safe (odd quote parity) even when it contains &; everything downstream uses
:: !REPO_DIR!.
set "REPO_DIR=%~dp0"


call :find_conda
if errorlevel 1 exit /b 1

:: --- first-time setup -------------------------------------------------------
set NEEDS_SETUP=0

call "!CONDA_PATH!" base
call conda env list | findstr /C:"patientgui" >nul 2>&1
if errorlevel 1 (
    echo [!] Conda environment 'patientgui' not found.
    set NEEDS_SETUP=1
)

if not exist "!REPO_DIR!fend\node_modules" (
    echo [!] Frontend dependencies not installed.
    set NEEDS_SETUP=1
)

if !NEEDS_SETUP!==1 (
    echo.
    echo First-time setup required. Do you want to run setup now?
    call :ask "Run setup"
    if errorlevel 1 (
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

:: --- simulation data --------------------------------------------------------
call :ensure_sim_data
if errorlevel 1 exit /b 1


:: --- refuse to launch quietly behind a stale backend ------------------------
call :check_port
if errorlevel 1 exit /b 1

:: --- launch -----------------------------------------------------------------
:: Build each command line in a variable first: nesting doubled quotes directly
:: inside `start "title" cmd /k "..."` breaks as soon as either path contains a
:: space. The !VAR! references keep a literal & intact (see the note at the top).
:: `start` builds the child's command line before it returns, so the endlocal at
:: the end of the script cannot pull these out from under it.
set "BACKEND_CMD=call "!CONDA_PATH!" patientgui && python -m server.simulate_server_cli "!REPO_DIR!sim\datafile1_filtered.npz" --port 8765 --pre-trigger 10 --loop -o "!REPO_DIR!output""
echo.
echo Starting backend server...
start "Backend Server" cmd /k "!BACKEND_CMD!"

set "FRONTEND_CMD=cd /d "!REPO_DIR!fend" && npm run dev"
echo Starting frontend dev server...
start "Frontend Dev Server" cmd /k "!FRONTEND_CMD!"

echo.
echo Both servers are starting in separate windows.
endlocal
exit /b 0


:: ===========================================================================
:: Subroutines
:: ===========================================================================

:ask
:: Yes/no prompt that fails SAFE. `(call )` clears ERRORLEVEL first, so a stale
:: value cannot be mistaken for an answer. Returns 0 for yes, 1 for anything
:: else - including choice not running at all (ERRORLEVEL 0) and EOF on stdin
:: (ERRORLEVEL 255). The caller treats 1 as "abort", so an unanswered prompt
:: never silently means yes.
(call )
choice /C YN /M "%~1"
if errorlevel 2 exit /b 1
if errorlevel 1 exit /b 0
exit /b 1


:find_conda
:: Probe the usual installation locations instead of hardcoding one. The three
:: launchers used to disagree about where conda lives, so on any given machine
:: at least one of them was dead.
set "CONDA_PATH="
if defined CONDA_OVERRIDE if exist "!CONDA_OVERRIDE!" set "CONDA_PATH=!CONDA_OVERRIDE!"
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
    echo           "%USERPROFILE%\miniconda3"
    echo           "%LOCALAPPDATA%\miniconda3"
    echo           "%USERPROFILE%\anaconda3"
    echo           "%LOCALAPPDATA%\anaconda3"
    echo           "%PROGRAMDATA%\miniconda3"
    echo.
    echo         If conda is installed somewhere else, set CONDA_OVERRIDE near
    echo         the top of this script to the full path of its activate.bat.
    pause
    exit /b 1
)
echo Using conda: !CONDA_PATH!
exit /b 0


:activate_env
:: Activate patientgui and REFUSE to continue if it did not take. Without this
:: check a failed activation leaves whatever python happens to be on PATH -
:: verified: a Store/Inkscape python answers the provenance question with
:: "unresolved" and would then receive the repair install.
:: %1 is OUR flag, not activate's - conda's activate.bat rejects a second
:: argument outright, so "quiet" only selects output redirection here.
if "%~1"=="quiet" (
    call "!CONDA_PATH!" patientgui >nul 2>&1
) else (
    call "!CONDA_PATH!" patientgui
)
if errorlevel 1 (
    echo.
    echo [ERROR] Could not activate the 'patientgui' conda environment.
    echo         Refusing to continue: any other python on PATH would answer
    echo         for it, and a repair would install into the wrong place.
    echo         Try: conda env list      to check the environment exists.
    pause
    exit /b 1
)
exit /b 0


:run_setup
:: Create the environment if it is missing.
call conda env list | findstr /C:"patientgui" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Creating conda environment from patientgui.yml...
    call conda env create -f "!REPO_DIR!bend\patientgui.yml"
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
call :activate_env
if errorlevel 1 exit /b 1
call pip install -e "!REPO_DIR!bend" --no-deps
if errorlevel 1 (
    echo [ERROR] Failed to install the backend package from
    echo         "!REPO_DIR!bend"
    pause
    exit /b 1
)
echo Backend package installed.

:: Install frontend dependencies. pushd/popd, because the old `cd /d` leaked
:: into the launch step's working directory.
if not exist "!REPO_DIR!fend\node_modules" (
    echo.
    echo Installing frontend dependencies...
    pushd "!REPO_DIR!fend"
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
call :activate_env quiet
if errorlevel 1 exit /b 1
python -c "import importlib.util as u,os,sys; s=u.find_spec('server'); p=(list(s.submodule_search_locations) or [None])[0] if s else None; g=os.path.normcase(os.path.abspath(os.path.dirname(p))) if p else ''; print('    backend package -> ' + (g or '(unresolved)')); sys.exit(0 if g==os.path.normcase(os.path.abspath(sys.argv[1])) else 1)" "!REPO_DIR!bend\src" 2>nul
if not errorlevel 1 goto :provenance_ok
echo.
echo [!] The 'server' package does NOT resolve to this checkout:
echo         this folder    "!REPO_DIR!bend\src"
echo     The backend would run a DIFFERENT copy of the code than the one in
echo     this folder, silently. This is usually a leftover editable install
echo     from another clone.
echo.
call :ask "Repair now by reinstalling the backend package from this folder"
if errorlevel 1 (
    echo.
    echo [ABORT] Not launching against a different checkout.
    pause
    exit /b 1
)
call pip install -e "!REPO_DIR!bend" --no-deps
if errorlevel 1 (
    echo [ERROR] Repair failed.
    pause
    exit /b 1
)
python -c "import importlib.util as u,os,sys; s=u.find_spec('server'); p=(list(s.submodule_search_locations) or [None])[0] if s else None; g=os.path.normcase(os.path.abspath(os.path.dirname(p))) if p else ''; print('    backend package -> ' + (g or '(unresolved)')); sys.exit(0 if g==os.path.normcase(os.path.abspath(sys.argv[1])) else 1)" "!REPO_DIR!bend\src" 2>nul
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
::
:: Filter carefully. Matching ":8765" anywhere hits an IPv6 address that merely
:: contains that hex group (fe80::1234:8765:... listening on 445) and the
:: foreign-address column of unrelated TIME_WAIT/ESTABLISHED rows. Requiring
:: LISTENING first and then a local address ending in ":8765" followed by
:: whitespace matches only a real listener. /R needs /C: here: without it
:: findstr splits the pattern on the space into two OR-patterns and the anchor
:: is lost.
netstat -ano -p TCP 2>nul | findstr /C:"LISTENING" | findstr /R /C:":8765 " >nul
if errorlevel 1 exit /b 0
echo.
echo [!] Port 8765 is already in use - a backend is probably still running.
echo     If you continue, the new backend will FAIL to bind and the frontend
echo     will silently connect to the old one instead.
echo.
echo     Close the previous "Backend Server" window first. To find the owner:
echo         netstat -ano ^| findstr :8765
echo.
call :ask "Continue anyway"
if errorlevel 1 (
    echo.
    echo [ABORT] Close the stale backend and run this again.
    pause
    exit /b 1
)
exit /b 0


:ensure_sim_data
:: The launcher has always pointed at this file and the repository has never
:: contained it, so simulation mode failed out of the box. Offer to synthesise
:: one instead of shipping a 30 MB binary.
set "SIM_DATA=!REPO_DIR!sim\datafile1_filtered.npz"
if exist "!SIM_DATA!" exit /b 0
echo.
echo [!] Simulation data not found:
echo         "!SIM_DATA!"
echo.
echo     One can be generated now: 64 channels, 60 s, 2000 Hz, about 30 MB.
echo     It is SYNTHETIC - band-limited noise with burst envelopes, not
echo     physiological data. Good for exercising the pipeline, useless for
echo     validating anything.
echo.
call :ask "Generate it now"
if errorlevel 1 (
    echo.
    echo [ABORT] The simulation server needs a data file to play back.
    echo         Generate one yourself with:
    echo             python -m server.make_sim_data "!SIM_DATA!"
    pause
    exit /b 1
)
if not exist "!REPO_DIR!sim" mkdir "!REPO_DIR!sim"
call :activate_env
if errorlevel 1 exit /b 1
python -m server.make_sim_data "!SIM_DATA!"
if errorlevel 1 (
    echo [ERROR] Failed to generate simulation data.
    pause
    exit /b 1
)
exit /b 0
