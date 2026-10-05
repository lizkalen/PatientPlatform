#!/usr/bin/env bash
# macOS/Linux launcher for PatientGUI - the counterpart of start.bat,
# start_sim.bat and start_quattrocento.bat, folded into one script:
#
#     ./start.sh sim            Simulation mode (no hardware needed)
#     ./start.sh live           Live mode - Ripple Trellis
#     ./start.sh quattrocento   Quattrocento / OTBioLab+ over TCP
#
# It performs the same checks as the .bat launchers: first-time setup,
# backend-package provenance, stale backend on the port, and (sim mode)
# missing simulation data. One difference: both servers run in THIS terminal
# instead of two new windows; Ctrl-C stops both.
#
# No `set -e`/`set -u` here: conda's shell hooks are not clean under either,
# and every step below checks its own outcome explicitly, .bat-style.

# ---------------------------------------------------------------------------
# If auto-detection below cannot find your conda installation, put the path
# of its BASE FOLDER here (the one containing etc/profile.d/conda.sh), e.g.
#   CONDA_OVERRIDE="$HOME/miniconda3"
# ---------------------------------------------------------------------------
CONDA_OVERRIDE=""

# Quattrocento only: path to your OTBioLab+ configuration file (extensionless
# XML / .otb+stp). It sets the channel count and sample rate deterministically.
# EDIT THIS, or export OTB_CONFIG before running.
OTB_CONFIG="${OTB_CONFIG:-$HOME/Desktop/test1.otb+stp}"

ENV_NAME=patientgui
PORT=8765

# This script's own folder. Everything downstream quotes it, so spaces, '&'
# and '!' in the checkout path are all fine (the pain the .bat files take to
# survive those is a cmd-only problem).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1

usage() {
    echo "Usage: $0 <mode>"
    echo
    echo "  sim            Simulation mode - plays back sim/datafile1_filtered.npz"
    echo "                 (offers to generate a synthetic file if it is missing)"
    echo "  live           Live mode - Ripple Trellis hardware via xipppy"
    echo "                 (Ripple ships xipppy only for Windows; this mode needs"
    echo "                 a macOS/Linux build of it installed by hand)"
    echo "  quattrocento   Quattrocento / OTBioLab+ over TCP"
    echo "                 (edit OTB_CONFIG near the top of this script first)"
    echo
    echo "Windows counterparts: start_sim.bat, start.bat, start_quattrocento.bat."
}

MODE="${1:-}"
case "$MODE" in
    sim|live|quattrocento) ;;
    *) usage; exit 1 ;;
esac


# ===========================================================================
# Helpers
# ===========================================================================

ask() {
    # Yes/no prompt that fails SAFE: only an explicit y/yes answers yes.
    # Anything else - including EOF when stdin is not a terminal - is "no",
    # so an unanswered prompt never silently means yes.
    local reply
    printf '%s [y/N] ' "$1"
    read -r reply || { echo; return 1; }
    case "$reply" in
        [Yy]|[Yy][Ee][Ss]) return 0 ;;
        *) return 1 ;;
    esac
}


find_conda() {
    # Probe the usual installation locations instead of hardcoding one, same
    # policy as the .bat launchers. A conda already configured in this shell
    # (CONDA_EXE is exported by `conda init` even with no environment active)
    # wins over the probe list.
    CONDA_BASE=""
    local candidates=() base
    [ -n "$CONDA_OVERRIDE" ] && candidates+=("$CONDA_OVERRIDE")
    [ -n "${CONDA_EXE:-}" ] && candidates+=("$(dirname "$(dirname "$CONDA_EXE")")")
    candidates+=(
        "$HOME/miniconda3"
        "$HOME/anaconda3"
        "$HOME/opt/miniconda3"
        "$HOME/opt/anaconda3"
        "$HOME/miniforge3"
        "$HOME/mambaforge"
        "/opt/miniconda3"
        "/opt/anaconda3"
        "/usr/local/miniconda3"
        "/usr/local/anaconda3"
        "/opt/homebrew/Caskroom/miniconda/base"
        "/opt/homebrew/Caskroom/miniforge/base"
        "/usr/local/Caskroom/miniconda/base"
    )
    for base in "${candidates[@]}"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$base"
            break
        fi
    done
    if [ -z "$CONDA_BASE" ]; then
        echo "[ERROR] Could not find a conda installation. Looked for"
        echo "        etc/profile.d/conda.sh under:"
        for base in "${candidates[@]}"; do
            echo "          $base"
        done
        echo
        echo "        If conda is installed somewhere else, set CONDA_OVERRIDE near"
        echo "        the top of this script to its base folder."
        return 1
    fi
    # shellcheck disable=SC1091
    if ! . "$CONDA_BASE/etc/profile.d/conda.sh"; then
        echo "[ERROR] Failed to source $CONDA_BASE/etc/profile.d/conda.sh"
        return 1
    fi
    echo "Using conda: $CONDA_BASE"
    return 0
}


activate_env() {
    # Activate patientgui and REFUSE to continue if it did not take. Without
    # this check a failed activation leaves whatever python happens to be on
    # PATH - which would answer the provenance question with "unresolved" and
    # then receive the repair install.
    if [ "${1:-}" = "quiet" ]; then
        conda activate "$ENV_NAME" >/dev/null 2>&1
    else
        conda activate "$ENV_NAME"
    fi
    if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
        echo
        echo "[ERROR] Could not activate the '$ENV_NAME' conda environment."
        echo "        Refusing to continue: any other python on PATH would answer"
        echo "        for it, and a repair would install into the wrong place."
        echo "        Try: conda env list      to check the environment exists."
        return 1
    fi
    return 0
}


have_env() {
    conda env list 2>/dev/null | grep -qE "^${ENV_NAME}[[:space:]]"
}


run_setup() {
    # Create the environment if it is missing. Note the platform-specific
    # spec: patientgui.yml is a win-64 export and cannot solve here.
    if ! have_env; then
        echo
        echo "Creating conda environment from patientgui.macos.yml..."
        if ! conda env create -f "$REPO_DIR/bend/patientgui.macos.yml"; then
            echo "[ERROR] Failed to create conda environment."
            return 1
        fi
        echo "Conda environment created successfully."
    fi

    # Install the backend package. Without this step a clean machine gets
    # "No module named server" from a backend that otherwise looks healthy.
    echo
    echo "Installing the backend package in editable mode..."
    activate_env || return 1
    if ! pip install -e "$REPO_DIR/bend" --no-deps; then
        echo "[ERROR] Failed to install the backend package from"
        echo "        $REPO_DIR/bend"
        return 1
    fi
    echo "Backend package installed."

    if [ ! -d "$REPO_DIR/fend/node_modules" ]; then
        echo
        echo "Installing frontend dependencies..."
        if ! (cd "$REPO_DIR/fend" && npm install); then
            echo "[ERROR] Failed to install npm dependencies."
            return 1
        fi
        echo "Frontend dependencies installed successfully."
    fi

    echo
    echo "Setup complete!"
    echo
    return 0
}


resolves_here() {
    # Does the editable install of `server` point at THIS checkout?
    python - "$REPO_DIR/bend/src" <<'PY'
import importlib.util as u, os, sys
s = u.find_spec('server')
p = (list(s.submodule_search_locations) or [None])[0] if s else None
g = os.path.normcase(os.path.abspath(os.path.dirname(p))) if p else ''
print('    backend package -> ' + (g or '(unresolved)'))
sys.exit(0 if g == os.path.normcase(os.path.abspath(sys.argv[1])) else 1)
PY
}


check_provenance() {
    # `server` is imported by name, so whichever editable install the
    # environment happens to carry wins - including one left behind by a
    # different clone of this repository. Launching that would run code
    # nobody is looking at.
    echo
    echo "Checking that the backend package points at this folder..."
    activate_env quiet || return 1
    if resolves_here; then
        echo "    OK - backend package resolves to this checkout."
        return 0
    fi
    echo
    echo "[!] The 'server' package does NOT resolve to this checkout:"
    echo "        this folder    $REPO_DIR/bend/src"
    echo "    The backend would run a DIFFERENT copy of the code than the one in"
    echo "    this folder, silently. This is usually a leftover editable install"
    echo "    from another clone."
    echo
    if ! ask "Repair now by reinstalling the backend package from this folder"; then
        echo
        echo "[ABORT] Not launching against a different checkout."
        return 1
    fi
    if ! pip install -e "$REPO_DIR/bend" --no-deps; then
        echo "[ERROR] Repair failed."
        return 1
    fi
    if ! resolves_here; then
        echo "[ERROR] Still resolving elsewhere after the repair. Check for a"
        echo "        conflicting .pth or an installed 'server' package in the"
        echo "        $ENV_NAME environment."
        return 1
    fi
    echo "    OK - backend package resolves to this checkout."
    return 0
}


check_port() {
    # A backend from an earlier run keeps the port. The new one then dies on
    # bind while the frontend silently connects to the OLD instance -
    # different config, possibly a half-open recording. lsof with an exact
    # port and -sTCP:LISTEN matches only a real listener, sidestepping the
    # substring pitfalls the .bat netstat filter has to fight.
    if ! command -v lsof >/dev/null 2>&1; then
        echo "[!] lsof not found - skipping the stale-backend port check."
        return 0
    fi
    if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        return 0
    fi
    echo
    echo "[!] Port $PORT is already in use - a backend is probably still running."
    echo "    If you continue, the new backend will FAIL to bind and the frontend"
    echo "    will silently connect to the old one instead."
    echo
    echo "    Stop the previous backend first. To find the owner:"
    echo "        lsof -nP -iTCP:$PORT -sTCP:LISTEN"
    echo
    if ! ask "Continue anyway"; then
        echo
        echo "[ABORT] Stop the stale backend and run this again."
        return 1
    fi
    return 0
}


ensure_sim_data() {
    SIM_DATA="$REPO_DIR/sim/datafile1_filtered.npz"
    [ -f "$SIM_DATA" ] && return 0
    echo
    echo "[!] Simulation data not found:"
    echo "        $SIM_DATA"
    echo
    echo "    One can be generated now: 64 channels, 60 s, 2000 Hz, about 30 MB."
    echo "    It is SYNTHETIC - band-limited noise with burst envelopes, not"
    echo "    physiological data. Good for exercising the pipeline, useless for"
    echo "    validating anything."
    echo
    if ! ask "Generate it now"; then
        echo
        echo "[ABORT] The simulation server needs a data file to play back."
        echo "        Generate one yourself with:"
        echo "            python -m server.make_sim_data \"$SIM_DATA\""
        return 1
    fi
    mkdir -p "$REPO_DIR/sim" || return 1
    activate_env || return 1
    if ! python -m server.make_sim_data "$SIM_DATA"; then
        echo "[ERROR] Failed to generate simulation data."
        return 1
    fi
    return 0
}


# ===========================================================================
# Main
# ===========================================================================

echo "Starting PatientGUI ($MODE)..."
echo

if [ "$MODE" = "quattrocento" ]; then
    echo "PREREQUISITE: OTBioLab+ must be running with its TCP/IP option enabled"
    echo "and acquisition started. This server connects to OTBioLab+ on"
    echo "127.0.0.1:31000, decodes the stream, and serves it to the frontend just"
    echo "like the Ripple backend."
    echo
fi

find_conda || exit 1

# --- first-time setup -------------------------------------------------------
NEEDS_SETUP=0
if ! have_env; then
    echo "[!] Conda environment '$ENV_NAME' not found."
    NEEDS_SETUP=1
fi
if [ ! -d "$REPO_DIR/fend/node_modules" ]; then
    echo "[!] Frontend dependencies not installed."
    NEEDS_SETUP=1
fi
if [ "$NEEDS_SETUP" = 1 ]; then
    echo
    echo "First-time setup required. Do you want to run setup now?"
    if ! ask "Run setup"; then
        echo
        echo "[ABORT] Setup declined. The platform cannot start without it -"
        echo "        the backend would fail immediately with \"No module named server\"."
        exit 1
    fi
    run_setup || exit 1
fi

# --- refuse to launch someone else's checkout -------------------------------
check_provenance || exit 1

# --- mode-specific preconditions --------------------------------------------
case "$MODE" in
    sim)
        ensure_sim_data || exit 1
        ;;
    live)
        # xipppy ships only as a win_amd64 wheel, so patientgui.macos.yml
        # cannot install it. Say so up front instead of letting the backend
        # die on ModuleNotFoundError in a wall of startup output.
        if ! python -c "import xipppy" >/dev/null 2>&1; then
            echo
            echo "[!] xipppy is not importable in the '$ENV_NAME' environment."
            echo "    Ripple ships xipppy only for Windows, so live mode cannot run"
            echo "    here unless you have installed a macOS/Linux build yourself."
            echo "    Use './start.sh sim' to test without hardware."
            echo
            if ! ask "Try to launch anyway"; then
                echo
                echo "[ABORT] Live mode needs xipppy."
                exit 1
            fi
        fi
        ;;
    quattrocento)
        if [ ! -f "$OTB_CONFIG" ]; then
            echo
            echo "[ERROR] OTBioLab+ configuration file not found:"
            echo "        $OTB_CONFIG"
            echo "        Edit OTB_CONFIG near the top of this script, or run:"
            echo "            OTB_CONFIG=\"/path/to/your/config.otb+stp\" $0 quattrocento"
            exit 1
        fi
        ;;
esac

# --- refuse to launch quietly behind a stale backend ------------------------
check_port || exit 1

# --- launch -----------------------------------------------------------------
case "$MODE" in
    sim)
        BACKEND_CMD=(python -m server.simulate_server_cli "$SIM_DATA" \
                     --port "$PORT" --pre-trigger 10 --loop -o "$REPO_DIR/output")
        ;;
    live)
        BACKEND_CMD=(python -m server.server_cli \
                     --port "$PORT" --pre-trigger 10 -o "$REPO_DIR/output")
        ;;
    quattrocento)
        BACKEND_CMD=(python -m server.quattrocento_server_cli --config "$OTB_CONFIG" \
                     --port "$PORT" --pre-trigger 10 -o "$REPO_DIR/output")
        ;;
esac

echo
echo "Starting backend server..."
"${BACKEND_CMD[@]}" &
BACKEND_PID=$!

cleanup() {
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# Catch a backend that dies on the spot (bad data file, missing module, port
# race) before burying its traceback under the frontend's output.
sleep 3
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo
    echo "[ERROR] The backend exited immediately - see the messages above."
    exit 1
fi

echo
echo "Both servers run in THIS terminal; Ctrl-C stops both."
echo "Open http://localhost:8080 in your browser once the frontend is up."
echo
echo "Starting frontend dev server..."
cd "$REPO_DIR/fend" && npm run dev
