#!/usr/bin/env bash
# Invoked by PowerShell. No apt, global pip, shell-profile or driver changes.
set -euo pipefail
probe_root=$(realpath "$1")
probe_policy=$2
probe_operation=$3
probe_seed=${4:-104729}
probe_seconds=${5:-180}
probe_condition=${6:-}
probe_world_slot=${7:-}
probe_task_priorities=${8:-}
if [ -n "$probe_task_priorities" ] && { [ "$probe_task_priorities" != priorities ] || [ "$probe_operation" != session ] || [ "$probe_policy" != steve1 ]; }; then
    echo 'Priority scheduling requires a STEVE-1 session.' >&2; exit 2
fi
case "$probe_policy" in vpt|steve1) ;; *) echo 'Unknown policy' >&2; exit 2;; esac
cd "$probe_root"
mkdir -p results .runtime .cache
export MAMBA_ROOT_PREFIX="$probe_root/.runtime/mamba"
export CONDA_PKGS_DIRS="$probe_root/.runtime/mamba/pkgs"
export PIP_CACHE_DIR="$probe_root/.cache/pip"
export XDG_CACHE_HOME="$probe_root/.cache"
probe_gradle_override="${GRADLE_USER_HOME:-}"
export GRADLE_USER_HOME="${GRADLE_USER_HOME:-$probe_root/.cache/gradle}"
export HF_HOME="$probe_root/.cache/huggingface"
export CUDA_CACHE_PATH="$probe_root/.cache/cuda"
# Local WSLg probe otherwise selected llvmpipe despite working D3D12 hardware.
# Set only this process tree; preserve an explicit caller renderer selection.
if [ -d /mnt/wslg ] && [ -e /usr/lib/wsl/lib/libd3d12.so ]; then
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
fi
if [ -e /usr/lib/wsl/lib/libcuda.so ]; then
    # cuDNN dlopens unversioned libcuda.so, absent from this WSL's ldconfig cache.
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
probe_mamba="$probe_root/.runtime/bin/micromamba"
probe_env="$probe_root/.runtime/$probe_policy"
if [ "$probe_operation" = setup ] || [ "$probe_operation" = resume-setup ]; then
    # Explicit setup command is the only path that installs dependencies.
    probe_setup_log="results/setup-$probe_policy-$(date -u +%Y%m%dT%H%M%S).log"
    # Forge's generated JAR cache is not safe for simultaneous policy builds.
    exec 9> "$probe_root/.runtime/setup.lock"
    if ! flock -n 9; then
        printf '%s\n' 'Another project setup is active; retry after it finishes.' | tee "$probe_setup_log" >&2
        exit 2
    fi
    exec > >(tee "$probe_setup_log") 2>&1
    if [ -z "$probe_gradle_override" ]; then
        export GRADLE_USER_HOME
        GRADLE_USER_HOME=$(mktemp -d -t minecraftrl-gradle-XXXXXXXX)
    fi
    if [ ! -x "$probe_mamba" ]; then
        probe_archive="$probe_root/.runtime/micromamba-2.0.5.tar.bz2"
        curl --fail --location --retry 2 'https://micro.mamba.pm/api/micromamba/linux-64/2.0.5' -o "$probe_archive"
        sha256sum "$probe_archive" > "$probe_archive.sha256"
        tar -xjf "$probe_archive" -C "$probe_root/.runtime" bin/micromamba
    fi
    if [ "$probe_operation" = setup ]; then
        if [ -d "$probe_env" ]; then
            echo "Environment exists; inspect logs before explicitly using resume-setup: $probe_env" >&2
            exit 2
        fi
        "$probe_mamba" create --yes --prefix "$probe_env" --file "configs/conda-$probe_policy-linux-64.candidate.txt"
    elif [ ! -x "$probe_env/bin/python" ]; then
        echo 'resume-setup requires an existing project-local Python environment.' >&2
        exit 2
    fi
    "$probe_mamba" run -p "$probe_env" python scripts/setup_dependencies.py --policy "$probe_policy"
    "$probe_mamba" env export -p "$probe_env" --explicit > "results/conda-$probe_policy-explicit.txt"
    "$probe_mamba" run -p "$probe_env" python -m pip freeze --all > "results/pip-$probe_policy-freeze.txt"
    echo 'Installed candidate environment; run preflight and real episodes before calling it validated.'
    exit 0
fi
if [ ! -x "$probe_mamba" ] || [ ! -d "$probe_env" ]; then
    echo 'Candidate environment missing. Run Get-OfficialSources.ps1 and -Operation setup first.' >&2
    exit 2
fi
if [ "$probe_operation" = download ]; then
    exec "$probe_mamba" run -p "$probe_env" python scripts/download_models.py --policy "$probe_policy"
fi
case "$probe_operation" in preflight|episode|suite|session|task-scenario|dialogue-scenario|world-scenario|semantic-world-scenario) ;; *) echo 'Unknown operation' >&2; exit 2;; esac
if [ -e results/STOP ]; then
    echo 'results/STOP is latched. Inspect prior shutdown logs, then remove this file to explicitly re-arm.' >&2
    exit 2
fi
if [ "$probe_operation" = world-scenario ] || [ "$probe_operation" = semantic-world-scenario ]; then
    if [ "$probe_policy" != steve1 ]; then echo 'World scenario requires STEVE-1' >&2; exit 2; fi
    probe_world_args=(--root "$probe_root")
    if [ "$probe_operation" = semantic-world-scenario ]; then probe_world_args+=(--semantic-fixture); fi
    exec "$probe_mamba" run -p "$probe_env" python scripts/run_world_scenario.py "${probe_world_args[@]}"
fi
if [ "$probe_operation" = task-scenario ] || [ "$probe_operation" = dialogue-scenario ]; then
    if [ "$probe_policy" != steve1 ]; then echo 'Task scenario requires STEVE-1' >&2; exit 2; fi
    probe_scenario_args=(--root "$probe_root" --seed "$probe_seed" --limit-s "$probe_seconds")
    if [ "$probe_operation" = dialogue-scenario ]; then probe_scenario_args+=(--dialogue); fi
    exec "$probe_mamba" run -p "$probe_env" python scripts/run_task_scenario.py "${probe_scenario_args[@]}"
fi
probe_args=(-m minecraftrl.cli "$probe_operation" --root "$probe_root" --policy "$probe_policy" --seed "$probe_seed" --limit-s "$probe_seconds" --stop-file "$probe_root/results/STOP")
if [ -n "$probe_condition" ]; then probe_args+=(--condition "$probe_condition"); fi
if [ -n "$probe_world_slot" ]; then probe_args+=(--world-slot "$probe_world_slot"); fi
if [ "$probe_task_priorities" = priorities ]; then probe_args+=(--task-priorities); fi
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null; then
    exec "$probe_mamba" run -p "$probe_env" xvfb-run -a python "${probe_args[@]}"
else
    exec "$probe_mamba" run -p "$probe_env" python "${probe_args[@]}"
fi
