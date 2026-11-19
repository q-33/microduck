#!/usr/bin/env bash
# setup_jetson.sh - set up an NVIDIA Jetson (Orin family, JetPack 7.2) to train this repo.
#
# Default is a DRY RUN: it prints every command it would run, in order, and changes nothing.
# A real run needs --apply, and only works on a Jetson that passes the preflight checks.
#
#   jetson/setup_jetson.sh             # dry run (always a dry run when not on a Jetson)
#   jetson/setup_jetson.sh --apply     # real run, on the Jetson only
#   jetson/setup_jetson.sh --help
#
# What it does, in order:
#   0. Preflight (read-only): Jetson? L4T/JetPack and CUDA versions supported? disk, RAM, swap;
#      repo and upstream submodule present at the pinned commit; uv.lock uses only allowed hosts.
#   1. System packages. Needs root. The script NEVER runs sudo itself: it prints the exact
#      apt-get commands for you to run, then stops. Re-run it afterwards.
#   2. uv (user install in ~/.local/bin, pinned version, no shell profile edits).
#   3. Python environment from upstream's uv.lock (`uv sync --locked`) in
#      third_party/microduck_rl/.venv.
#   4. CUDA 13 PyTorch and CUDA runtime wheels from PyTorch's own index, pinned, --no-deps.
#   5. Verification: imports, torch sees CUDA and runs a CUDA matmul, then two 64-environment,
#      5-iteration GPU training smoke tests (upstream task and our replica task).
#
# Safety rules this script keeps:
#   - No sudo password is ever read, stored or piped. Root commands are printed, not run.
#   - Nothing it did not create in this run is deleted or overwritten. If something is in the
#     way it stops and says what. Shell profiles (~/.bashrc etc.) are not edited.
#   - Only official sources: the device's own apt repos, astral.sh (uv), PyPI, PyTorch's index,
#     and github.com for the git dependency pinned in upstream's uv.lock.
#   - Every step checks whether it is already done, so running it twice is safe.
#
# Test hooks (used only by tests on a non-Jetson machine; setting any of them forces a dry run):
#   SETUP_JETSON_TEST_ROOT   folder that stands in for / when reading device facts
#   SETUP_JETSON_TEST_UNAME  "<kernel> <machine>", for example "Linux aarch64"

set -euo pipefail

# ======================================================================================
# Settings. Every version, source and URL the script uses is here.
# ======================================================================================

# --- Supported platform -----------------------------------------------------------------
# Upstream's verified platform (third_party/microduck_rl/microduck_jetson_startup.md, section 0.1,
# lines 14-26): Jetson Orin NX 16GB, Ubuntu 24.04 aarch64, JetPack 7.2, L4T R39.2, system CUDA 13.2,
# Python 3.12, PyTorch 2.9.1+cu130, MuJoCo 3.10.0, Warp 1.12.0.
# NVIDIA: JetPack 7.2 = Jetson Linux 39.2, CUDA 13.2.1, L4T Ubuntu 24.04, "Jetson Orin Family"
#   https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.md
#   https://developer.nvidia.com/embedded/jetpack-archive
# Space-separated L4T versions (major.minor.patch) the GPU steps below are known to work on.
SUPPORTED_L4T_VERSIONS="39.2.0"     # VERIFIED by upstream (as "R39.2"), on an Orin NX, not a Nano
# JetPack 7.2.1 / L4T 39.2.1 (CUDA 13.2.2) also supports Orin but upstream never ran it:
#   https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2.1.md
# UNVERIFIED: add "39.2.1" above only after deciding to accept that risk.
SUPPORTED_CUDA_PREFIX="13.2"        # VERIFIED (NVIDIA: 13.2.1 in JetPack 7.2; upstream: 13.2)
REQUIRED_MODEL_WORD="Orin"          # VERIFIED: torch cu130 build ran on Orin (sm_87) upstream
EXPECTED_MODEL_WORDS="Orin Nano"    # UNVERIFIED: upstream never ran on an Orin Nano; only warns
MIN_FREE_DISK_GB=25                 # upstream startup.md line 34 ("at least 25GB")

# --- Repo -------------------------------------------------------------------------------
UPSTREAM_REL_DIR="third_party/microduck_rl"
UPSTREAM_PINNED_COMMIT="25f762409c526bed7a1fabd60a796d05a9ada873"

# --- System packages (root) -------------------------------------------------------------
# Same list as upstream's verified script (deploy_microduck_jetson.sh lines 35-39).
# Installed from the apt repos already configured on the device; no repo is added.
APT_PACKAGES="git curl wget build-essential tmux ffmpeg libgl1 libgles2 libegl1 libglfw3 \
libglfw3-dev libudev-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
libgstreamer-plugins-bad1.0-dev"

# --- uv ---------------------------------------------------------------------------------
# Pinned to the uv used for the Mac environment. The versioned installer redirects to
# releases.astral.sh and downloads uv from there, falling back to github.com/astral-sh/uv.
# UV_NO_MODIFY_PATH=1 stops it editing shell profiles (installer source, 0.13.9).
UV_VERSION="0.13.9"
UV_INSTALLER_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"

# --- Python environment (from upstream's uv.lock) ---------------------------------------
# uv.lock pins every package to PyPI, plus one git dependency (Rhoban/bam at a fixed commit).
PYPI_INDEX_URL="https://pypi.org/simple"
PYPI_FILES_URL="https://files.pythonhosted.org"
GIT_DEPENDENCY_URL="https://github.com/Rhoban/bam.git"
# Every host uv.lock may reference; preflight stops if it finds any other.
LOCK_ALLOWED_HOSTS="https://pypi.org https://files.pythonhosted.org https://github.com"
# Use Ubuntu 24.04's own python3.12; never download a Python build.
# UNVERIFIED: upstream does not say which 3.12 it used; with a system 3.12 present, uv's default
# also picks it rather than downloading one.
UV_PYTHON_PREFERENCE_VALUE="only-system"
UV_PYTHON_DOWNLOADS_VALUE="never"
UV_HTTP_TIMEOUT_VALUE=600           # upstream deploy script line 7 (CUDA wheels are large)

# --- CUDA PyTorch (replaces the CPU torch that uv.lock installs on aarch64) ---------------
# Pins and order from upstream deploy script lines 65-85 / startup.md section 0.11.
# All 19 wheels checked present for aarch64 on https://download.pytorch.org/whl/cu130/<name>/
# (2026-11-18). VERIFIED to exist; VERIFIED working by upstream on an Orin NX only.
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
TORCH_PINS="torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1"
TORCH_EXPECTED_VERSION="2.9.1+cu130"
CUDA_WHEEL_PINS="nvidia-cuda-nvrtc==13.0.48 nvidia-cuda-runtime==13.0.48 \
nvidia-cuda-cupti==13.0.48 nvidia-cudnn-cu13==9.13.0.50 nvidia-cublas==13.0.0.19 \
nvidia-cufft==12.0.0.15 nvidia-curand==10.4.0.35 nvidia-cusolver==12.0.3.29 \
nvidia-cusparse==12.6.2.49 nvidia-cusparselt-cu13==0.8.0 nvidia-nccl-cu13==2.27.7 \
nvidia-nvshmem-cu13==3.3.24 nvidia-nvtx==13.0.39 nvidia-nvjitlink==13.0.39 \
nvidia-cufile==1.15.0.42 triton==3.5.1"

# --- Smoke tests ------------------------------------------------------------------------
# From README.md "Training smoke test" and duck_train/train.py docstring, on the GPU
# (no --gpu-ids None), 64 environments, 5 iterations.
SMOKE_NUM_ENVS=64
SMOKE_ITERATIONS=5
SMOKE_UPSTREAM_TASK="Mjlab-Velocity-Flat-MicroDuck"
SMOKE_REPLICA_TASK="Mjlab-Velocity-Flat-MicroDuck-Replica"

# Runtime environment (upstream deploy script lines 7-10, startup.md 0.6).
CUDA_HOME_DIR="/usr/local/cuda"
MUJOCO_GL_VALUE="egl"

# ======================================================================================
# Helpers
# ======================================================================================

EXIT_USAGE=2      # bad arguments, or --apply refused
EXIT_PREFLIGHT=3  # preflight found something unsupported or in the way
EXIT_ROOT=4       # root step needed: run the printed commands, then re-run

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   WARNING: %s\n' "$*"; }
die()  { local code="$1"; shift; printf '\nSTOP: %s\n' "$*" >&2; exit "$code"; }

# Shell-quote arguments for display.
q() {
  local out="" a
  for a in "$@"; do out="${out}$(printf '%q' "$a") "; done
  printf '%s' "${out% }"
}

# run [--in DIR] CMD...   Print the command; execute it only with --apply.
run() {
  local dir=""
  if [ "$1" = "--in" ]; then dir="$2"; shift 2; fi
  if [ -n "$dir" ]; then
    say "   \$ (cd $(q "$dir") && $(q "$@"))"
  else
    say "   \$ $(q "$@")"
  fi
  if [ "$MODE" = "apply" ]; then
    if [ -n "$dir" ]; then (cd "$dir" && "$@"); else "$@"; fi
  fi
}

# Path of a device-fact file, honouring the test root.
fact_path() { printf '%s%s' "${SETUP_JETSON_TEST_ROOT:-}" "$1"; }

usage() {
  sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
}

# ======================================================================================
# Arguments and mode
# ======================================================================================

MODE="dry-run"
WANT_APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) WANT_APPLY=1 ;;
    --dry-run) WANT_APPLY=0 ;;
    -h|--help) usage; exit 0 ;;
    *) say "Unknown option: $arg (see --help)" >&2; exit "$EXIT_USAGE" ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
UP="$REPO/$UPSTREAM_REL_DIR"
VENV="$UP/.venv"
VENV_PY="$VENV/bin/python"
STATE_DIR="$REPO/out/jetson_setup"                 # out/ is git-ignored
VENV_MARKER="$STATE_DIR/venv-created-by-setup_jetson"
SMOKE_DIR="$REPO/out/jetson_smoke"

TEST_MODE=0
if [ -n "${SETUP_JETSON_TEST_ROOT:-}" ] || [ -n "${SETUP_JETSON_TEST_UNAME:-}" ]; then
  TEST_MODE=1
fi

say "MicroDuck Jetson setup"
say "Repo: $REPO"

# ======================================================================================
# 0. Preflight (read-only)
# ======================================================================================

step "0. Preflight (read-only)"

if [ -n "${SETUP_JETSON_TEST_UNAME:-}" ]; then
  KERNEL="${SETUP_JETSON_TEST_UNAME%% *}"
  MACHINE="${SETUP_JETSON_TEST_UNAME#* }"
else
  KERNEL="$(uname -s)"
  MACHINE="$(uname -m)"
fi
info "System: $KERNEL $MACHINE"

IS_JETSON=0
TEGRA_FILE="$(fact_path /etc/nv_tegra_release)"
if [ "$KERNEL" = "Linux" ] && [ "$MACHINE" = "aarch64" ] && [ -r "$TEGRA_FILE" ]; then
  IS_JETSON=1
fi

if [ "$TEST_MODE" = 1 ]; then
  info "TEST MODE: device facts come from test hooks; a real run is not possible."
fi

if [ "$IS_JETSON" = 0 ]; then
  info "Not a Jetson (needs Linux aarch64 with /etc/nv_tegra_release)."
  if [ "$WANT_APPLY" = 1 ]; then
    die "$EXIT_USAGE" "--apply refused: this is not a Jetson. Nothing was changed."
  fi
  info "Dry run only. Device checks are skipped; the plan below assumes a supported Jetson."
else
  # --- L4T / JetPack ---
  # First line looks like: "# R39 (release), REVISION: 2.0, GCID: ..., BOARD: ..., EABI: aarch64, ..."
  tegra_line="$(head -n 1 "$TEGRA_FILE")"
  l4t_major="$(printf '%s' "$tegra_line" | sed -n 's/^# R\([0-9][0-9]*\) (release).*/\1/p')"
  l4t_rev="$(printf '%s' "$tegra_line" | sed -n 's/.*REVISION: \([0-9][0-9.]*\).*/\1/p')"
  [ -n "$l4t_major" ] && [ -n "$l4t_rev" ] \
    || die "$EXIT_PREFLIGHT" "cannot read the L4T version from $TEGRA_FILE (first line: '$tegra_line')."
  case "$l4t_rev" in *.*.*) ;; *.*) ;; *) l4t_rev="${l4t_rev}.0" ;; esac
  L4T="${l4t_major}.${l4t_rev}"
  info "L4T (Jetson Linux): $L4T"

  jetpack_pkg="unknown (nvidia-jetpack package not installed)"
  if [ "$TEST_MODE" = 0 ] && command -v dpkg-query >/dev/null 2>&1; then
    v="$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || true)"
    [ -n "$v" ] && jetpack_pkg="$v"
  fi
  info "JetPack package: $jetpack_pkg"

  # --- Board ---
  MODEL_FILE="$(fact_path /proc/device-tree/model)"
  MODEL="unknown"
  [ -r "$MODEL_FILE" ] && MODEL="$(tr -d '\0' < "$MODEL_FILE")"
  info "Board: $MODEL"

  # --- CUDA ---
  CUDA_VERSION=""
  CUDA_JSON="$(fact_path "$CUDA_HOME_DIR/version.json")"
  if [ -r "$CUDA_JSON" ]; then
    # "cuda" : { "name" : "CUDA SDK", "version" : "13.2.1" }
    CUDA_VERSION="$(sed -n '/"cuda"[[:space:]]*:/,/}/s/.*"version"[[:space:]]*:[[:space:]]*"\([0-9.]*\)".*/\1/p' \
      "$CUDA_JSON" | head -n 1)"
  fi
  if [ -z "$CUDA_VERSION" ] && [ "$TEST_MODE" = 0 ] && [ -x "$CUDA_HOME_DIR/bin/nvcc" ]; then
    CUDA_VERSION="$("$CUDA_HOME_DIR/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p')"
  fi
  info "CUDA toolkit: ${CUDA_VERSION:-not found}"

  # --- Version gate ---
  l4t_ok=0
  for s in $SUPPORTED_L4T_VERSIONS; do [ "$s" = "$L4T" ] && l4t_ok=1; done
  if [ "$l4t_ok" = 0 ]; then
    die "$EXIT_PREFLIGHT" "unsupported Jetson Linux version.
  Found:    L4T $L4T (JetPack package: $jetpack_pkg), CUDA ${CUDA_VERSION:-not found}
  Expected: L4T one of [$SUPPORTED_L4T_VERSIONS] (JetPack 7.2) with CUDA $SUPPORTED_CUDA_PREFIX.x
  The GPU install steps (PyTorch 2.9.1+cu130 and CUDA 13 wheels) are only known to work there.
  Flash JetPack 7.2, or, after checking, add your L4T version to SUPPORTED_L4T_VERSIONS at the
  top of this script. Nothing was changed."
  fi
  case "$CUDA_VERSION" in
    "$SUPPORTED_CUDA_PREFIX"|"$SUPPORTED_CUDA_PREFIX".*) ;;
    *) die "$EXIT_PREFLIGHT" "unsupported or missing CUDA toolkit.
  Found:    CUDA '${CUDA_VERSION:-not found}' at $CUDA_HOME_DIR (L4T $L4T)
  Expected: CUDA $SUPPORTED_CUDA_PREFIX.x, as shipped with JetPack 7.2. Nothing was changed." ;;
  esac
  case "$MODEL" in
    *"$REQUIRED_MODEL_WORD"*) ;;
    *) die "$EXIT_PREFLIGHT" "board '$MODEL' is not a Jetson $REQUIRED_MODEL_WORD; the pinned PyTorch build is
  only known to work on Orin (sm_87). Nothing was changed." ;;
  esac
  case "$MODEL" in
    *"$EXPECTED_MODEL_WORDS"*) ;;
    *) warn "board is not an '$EXPECTED_MODEL_WORDS'; continuing (Orin family is supported)." ;;
  esac
  info "Versions supported: L4T $L4T, CUDA $CUDA_VERSION."

  # --- Memory and swap ---
  MEMINFO="$(fact_path /proc/meminfo)"
  if [ -r "$MEMINFO" ]; then
    mem_kb="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' "$MEMINFO")"
    swap_kb="$(sed -n 's/^SwapTotal:[[:space:]]*\([0-9]*\).*/\1/p' "$MEMINFO")"
    info "RAM: $((${mem_kb:-0} / 1024)) MiB, swap: $((${swap_kb:-0} / 1024)) MiB"
    if [ "${swap_kb:-0}" = 0 ]; then
      warn "no swap. RAM is shared with the GPU on Orin; large runs may be killed for memory."
    fi
  fi

  # --- Python 3.12 for uv (only-system) ---
  if [ "$TEST_MODE" = 0 ]; then
    if command -v python3.12 >/dev/null 2>&1; then
      info "python3.12: $(command -v python3.12)"
    else
      die "$EXIT_PREFLIGHT" "python3.12 not found. Ubuntu 24.04 (JetPack 7.2) ships it; this script does
  not download a Python build. Nothing was changed."
    fi
  fi
fi

# --- Disk (checked on every machine: it is the repo's own disk) ---
free_kb="$(df -Pk "$REPO" | awk 'NR==2 {print $4}')"
free_gb=$((free_kb / 1024 / 1024))
info "Free disk at the repo: ${free_gb} GB (need ${MIN_FREE_DISK_GB} GB)"
if [ "$free_gb" -lt "$MIN_FREE_DISK_GB" ]; then
  if [ "$IS_JETSON" = 1 ]; then
    die "$EXIT_PREFLIGHT" "only ${free_gb} GB free at $REPO; upstream asks for at least ${MIN_FREE_DISK_GB} GB."
  fi
  warn "less than ${MIN_FREE_DISK_GB} GB free (only matters on the Jetson)."
fi

# --- Repo and submodule at the pinned commit ---
command -v git >/dev/null 2>&1 || die "$EXIT_PREFLIGHT" "git not found. Install it (sudo apt-get install git), then re-run."
[ -f "$UP/pyproject.toml" ] && [ -f "$UP/uv.lock" ] || die "$EXIT_PREFLIGHT" "upstream submodule is missing at $UP.
  Run this yourself, then re-run:  git -C $(q "$REPO") submodule update --init"
pinned_in_repo="$(git -C "$REPO" ls-tree HEAD "$UPSTREAM_REL_DIR" | awk '{print $3}')"
checked_out="$(git -C "$UP" rev-parse HEAD)"
info "Submodule: pinned in repo ${pinned_in_repo:-none}, checked out $checked_out"
[ "$pinned_in_repo" = "$UPSTREAM_PINNED_COMMIT" ] || die "$EXIT_PREFLIGHT" "the repo pins upstream at '${pinned_in_repo:-none}',
  but this script was written for $UPSTREAM_PINNED_COMMIT. Update UPSTREAM_PINNED_COMMIT only after
  checking the new commit's Jetson notes."
[ "$checked_out" = "$UPSTREAM_PINNED_COMMIT" ] || die "$EXIT_PREFLIGHT" "the submodule is checked out at $checked_out,
  not the pinned $UPSTREAM_PINNED_COMMIT. Run this yourself, then re-run:
  git -C $(q "$REPO") submodule update --init"
if [ -n "$(git -C "$UP" status --porcelain --untracked-files=no)" ]; then
  die "$EXIT_PREFLIGHT" "the upstream submodule has local changes (git -C $UP status). It must be unmodified."
fi

# --- uv.lock hosts ---
lock_hosts="$(grep -oE 'https?://[A-Za-z0-9.-]+' "$UP/uv.lock" | sort -u | tr '\n' ' ')"
info "Hosts in uv.lock: $lock_hosts"
for h in $lock_hosts; do
  ok=0
  for a in $LOCK_ALLOWED_HOSTS; do [ "$h" = "$a" ] && ok=1; done
  [ "$ok" = 1 ] || die "$EXIT_PREFLIGHT" "uv.lock references '$h', which is not in LOCK_ALLOWED_HOSTS."
done

if [ "$WANT_APPLY" = 1 ]; then
  if [ "$TEST_MODE" = 1 ]; then
    die "$EXIT_USAGE" "--apply refused: test hooks (SETUP_JETSON_TEST_*) are set. Nothing was changed."
  fi
  MODE="apply"
fi
# Only on a real Jetson are the "already done?" checks meaningful.
CHECK_STATE=0
[ "$IS_JETSON" = 1 ] && [ "$TEST_MODE" = 0 ] && CHECK_STATE=1

info "Preflight passed."
if [ "$MODE" = "apply" ]; then
  say ""; say "MODE: APPLY. Commands below are run."
else
  say ""; say "MODE: DRY RUN. Commands below are printed only; nothing is changed or downloaded."
fi

# Environment for every later command (not written to any shell profile).
export PATH="$CUDA_HOME_DIR/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export UV_HTTP_TIMEOUT="$UV_HTTP_TIMEOUT_VALUE"
export UV_PYTHON_PREFERENCE="$UV_PYTHON_PREFERENCE_VALUE"
export UV_PYTHON_DOWNLOADS="$UV_PYTHON_DOWNLOADS_VALUE"
export MUJOCO_GL="$MUJOCO_GL_VALUE"
say "Environment for this run: PATH+=$CUDA_HOME_DIR/bin:\$HOME/.local/bin, LD_LIBRARY_PATH+=$CUDA_HOME_DIR/lib64,"
say "  UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT, UV_PYTHON_PREFERENCE=$UV_PYTHON_PREFERENCE,"
say "  UV_PYTHON_DOWNLOADS=$UV_PYTHON_DOWNLOADS, MUJOCO_GL=$MUJOCO_GL"

# ======================================================================================
# 1. System packages (root: printed, never run by this script)
# ======================================================================================

step "1. System packages (needs root; you run these yourself)"
missing=""
if [ "$CHECK_STATE" = 1 ]; then
  for p in $APT_PACKAGES; do
    st="$(dpkg-query -W -f='${Status}' "$p" 2>/dev/null || true)"
    [ "$st" = "install ok installed" ] || missing="$missing $p"
  done
  missing="${missing# }"
else
  missing="$(printf '%s' "$APT_PACKAGES" | tr -s ' ')"
fi
if [ -z "$missing" ]; then
  info "Already done: all packages installed."
else
  info "Missing: $missing"
  say "   \$ sudo apt-get update"
  say "   \$ sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y $missing"
  if [ "$MODE" = "apply" ]; then
    say ""
    say "Run the two commands above yourself (sudo asks for your password), then re-run:"
    say "   $(q "$0") --apply"
    exit "$EXIT_ROOT"
  fi
  info "(A real run stops here until these are installed.)"
fi

# ======================================================================================
# 2. uv
# ======================================================================================

step "2. uv $UV_VERSION (user install in ~/.local/bin)"
UV_BIN=""
if [ "$CHECK_STATE" = 1 ]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  elif [ -e "$HOME/.local/bin/uv" ] || [ -e "$HOME/.local/bin/uvx" ]; then
    die "$EXIT_PREFLIGHT" "$HOME/.local/bin/uv or uvx exists but does not run. It is in the way; this
  script will not overwrite it. Check or move it yourself, then re-run."
  fi
fi
if [ -n "$UV_BIN" ]; then
  info "Already done: $UV_BIN ($("$UV_BIN" --version))"
else
  if [ "$MODE" = "apply" ]; then
    mkdir -p "$STATE_DIR"
    installer="$(mktemp "$STATE_DIR/uv-installer.XXXXXX")"   # created by this run
  else
    installer="$STATE_DIR/uv-installer.XXXXXX"
    say "   \$ mkdir -p $(q "$STATE_DIR")"
  fi
  run curl --proto '=https' --tlsv1.2 -LsSf -o "$installer" "$UV_INSTALLER_URL"
  run env UV_NO_MODIFY_PATH=1 sh "$installer"
  if [ "$MODE" = "apply" ]; then
    rm -f -- "$installer"   # the temporary file this run created, nothing else
  fi
  UV_BIN="$HOME/.local/bin/uv"
  run "$UV_BIN" --version
fi
[ -n "$UV_BIN" ] || UV_BIN="uv"

# ======================================================================================
# 3. Python environment from upstream's uv.lock
# ======================================================================================

step "3. Python environment ($VENV)"
torch_ok() {
  [ -x "$VENV_PY" ] && [ "$("$VENV_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)" \
    = "$TORCH_EXPECTED_VERSION" ]
}
VENV_OURS=1        # may this run change the environment?
NEED_SYNC=1
if [ "$CHECK_STATE" = 1 ] && [ -e "$VENV" ]; then
  if [ -f "$VENV_MARKER" ]; then
    info "Environment exists and was created by this script earlier."
  else
    VENV_OURS=0
  fi
  if torch_ok; then
    NEED_SYNC=0
    info "Already done: environment has torch $TORCH_EXPECTED_VERSION (re-syncing would remove it)."
  elif [ "$VENV_OURS" = 0 ]; then
    die "$EXIT_PREFLIGHT" "an environment already exists at $VENV that this script did not create,
  and it does not have torch $TORCH_EXPECTED_VERSION. It will not be changed. Move it aside
  yourself (for example: mv $(q "$VENV") $(q "$VENV.old")), then re-run."
  fi
fi
if [ "$NEED_SYNC" = 1 ]; then
  info "Downloads (as pinned in uv.lock): $PYPI_INDEX_URL, $PYPI_FILES_URL, $GIT_DEPENDENCY_URL"
  # Record ownership BEFORE creating the environment, so a sync that fails halfway can be retried.
  if [ ! -e "$VENV" ]; then
    if [ "$MODE" = "apply" ]; then
      mkdir -p "$STATE_DIR"
      printf 'created %s for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VENV" > "$VENV_MARKER"
    else
      say "   (first record that this script creates it: $VENV_MARKER)"
    fi
  fi
  # --locked: install exactly uv.lock and never rewrite it (nothing under third_party/ is edited).
  run --in "$UP" "$UV_BIN" sync --locked
fi

# ======================================================================================
# 4. CUDA PyTorch and CUDA 13 runtime wheels (pinned, --no-deps so NumPy etc. stay locked)
# ======================================================================================

step "4. CUDA PyTorch $TORCH_EXPECTED_VERSION and CUDA 13 wheels from $PYTORCH_INDEX_URL"
if [ "$CHECK_STATE" = 1 ] && [ "$NEED_SYNC" = 0 ] && torch_ok; then
  info "Already done: torch $TORCH_EXPECTED_VERSION."
else
  # shellcheck disable=SC2086  # word splitting of the pin lists is intended
  run "$UV_BIN" pip install --python "$VENV_PY" --reinstall --no-deps $TORCH_PINS \
    --index-url "$PYTORCH_INDEX_URL"
fi
missing_wheels="$CUDA_WHEEL_PINS"
if [ "$CHECK_STATE" = 1 ] && [ -x "$VENV_PY" ]; then
  frozen="$("$UV_BIN" pip freeze --python "$VENV_PY" 2>/dev/null || true)"
  missing_wheels=""
  for w in $CUDA_WHEEL_PINS; do
    printf '%s\n' "$frozen" | grep -qixF -- "$w" || missing_wheels="$missing_wheels $w"
  done
  missing_wheels="${missing_wheels# }"
fi
if [ -z "$missing_wheels" ]; then
  info "Already done: all CUDA 13 wheels at the pinned versions."
else
  if [ "$CHECK_STATE" = 1 ] && [ "$VENV_OURS" = 0 ]; then
    die "$EXIT_PREFLIGHT" "the environment at $VENV was not created by this script and lacks:
  $missing_wheels
  It will not be changed. Move it aside yourself, then re-run."
  fi
  # shellcheck disable=SC2086
  run "$UV_BIN" pip install --python "$VENV_PY" --no-deps $missing_wheels \
    --index-url "$PYTORCH_INDEX_URL"
fi

# ======================================================================================
# 5. Verification
# ======================================================================================

step "5. Verification"
# Checks from upstream startup.md section 0.12: a real CUDA operation, not only is_available().
VERIFY_PY='import mediapy, mujoco, numpy, torch, warp, mjlab_microduck
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.__version__ == "'"$TORCH_EXPECTED_VERSION"'", "wrong torch build"
assert torch.cuda.is_available(), "torch does not see CUDA"
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
print("NumPy:", numpy.__version__, "MuJoCo:", mujoco.__version__, "Warp:", warp.__version__)
a = torch.randn(512, 512, device="cuda"); b = torch.randn(512, 512, device="cuda")
c = torch.matmul(a, b); torch.cuda.synchronize()
print("CUDA matmul OK:", c.device, tuple(c.shape))
print("Project import: OK")'
say "   \$ (cd $(q "$UP") && $(q "$UV_BIN") run --no-sync python - <<'PY'"
printf '%s\n' "$VERIFY_PY" | sed 's/^/     /'
say "   PY)"
if [ "$MODE" = "apply" ]; then
  (cd "$UP" && "$UV_BIN" run --no-sync python -c "$VERIFY_PY")
fi

if [ "$MODE" = "apply" ]; then
  mkdir -p "$SMOKE_DIR/upstream" "$SMOKE_DIR/replica"
else
  say "   \$ mkdir -p $(q "$SMOKE_DIR/upstream") $(q "$SMOKE_DIR/replica")"
fi
info "First run compiles MuJoCo Warp CUDA kernels; this can take several minutes."
# Logs go to logs/ under the current folder (README), so each test runs in its own folder.
run --in "$SMOKE_DIR/upstream" "$UV_BIN" run --no-sync --project "$UP" train "$SMOKE_UPSTREAM_TASK" \
  --env.scene.num-envs "$SMOKE_NUM_ENVS" --agent.max-iterations "$SMOKE_ITERATIONS" \
  --agent.logger tensorboard
run --in "$SMOKE_DIR/replica" "$UV_BIN" run --no-sync --project "$UP" python "$REPO/duck_train/train.py" \
  "$SMOKE_REPLICA_TASK" --env.scene.num-envs "$SMOKE_NUM_ENVS" \
  --agent.max-iterations "$SMOKE_ITERATIONS" --agent.logger tensorboard

# ======================================================================================
# Done
# ======================================================================================

step "Notes"
say "   Always use 'uv run --no-sync' on the Jetson. A plain 'uv run' or 'uv sync' puts the CPU torch"
say "   from uv.lock back; if that happens, re-run this script."
say "   For your own shells, add these lines to ~/.bashrc yourself (this script does not edit it):"
say "     export PATH=$CUDA_HOME_DIR/bin:\$HOME/.local/bin:\$PATH"
say "     export LD_LIBRARY_PATH=$CUDA_HOME_DIR/lib64:\$LD_LIBRARY_PATH"
say "     export UV_HTTP_TIMEOUT=$UV_HTTP_TIMEOUT_VALUE"
say "     export MUJOCO_GL=$MUJOCO_GL_VALUE"
say "   Power mode is not changed by this script. Check it with: sudo nvpmodel -q"
say "   Mode numbers differ between Jetson models; pick one from that board's own list."
if [ "$MODE" = "apply" ]; then
  say ""; say "Setup finished. Smoke-test logs: $SMOKE_DIR"
else
  say ""; say "Dry run finished. Nothing was changed. On the Jetson, run with --apply."
fi
