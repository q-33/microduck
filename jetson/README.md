# Jetson setup

`setup_jetson.sh` gets an NVIDIA Jetson Orin ready to train this repo's walking tasks on its GPU.
We wrote our own instead of using upstream's `deploy_microduck_jetson.sh`, because that one
downloads through a third-party GitHub mirror, reads the sudo password from an environment
variable, and can delete its target folder.

## What it needs

- A Jetson from the Orin family running **JetPack 7.2** (Jetson Linux 39.2, CUDA 13.2). That is
  the only version upstream has tested the GPU install on (on an Orin NX 16GB). The Orin Nano
  developer kit is flashed from NVIDIA's ISO image on a USB stick; JetPack 7.2 has no SD card image.
- This repo cloned on the Jetson, with the upstream submodule:
  `git submodule update --init`
- At least 25 GB free disk and an internet connection (several GB of downloads).

## How to run it

```bash
jetson/setup_jetson.sh            # dry run: prints every command, changes nothing
jetson/setup_jetson.sh --apply    # real run, on the Jetson only
```

It checks the board and versions first. If anything is unexpected (another JetPack, another CUDA,
a modified submodule, a Python environment it did not create) it stops, says what it found, and
changes nothing.

Installing system packages needs root. The script never runs `sudo` and never asks for a password.
It prints the two `apt-get` commands for you to run, then stops. Run them, then run the script
again. It skips whatever is already done, so running it again is always safe.

At the end it checks that PyTorch sees the GPU and runs two short GPU training tests
(64 simulations, 5 iterations): upstream's walking task and our replica task. Logs go to
`out/jetson_smoke/`.

## After setup

- Always use `uv run --no-sync` on the Jetson. A plain `uv run` or `uv sync` replaces the GPU
  PyTorch with the CPU one from the lock file. If that happens, run the script again.
- The script does not edit `~/.bashrc`. It prints four `export` lines to add yourself.
- It does not change the power mode. Check it with `sudo nvpmodel -q`.

## Another JetPack version

The supported versions are variables at the top of the script (`SUPPORTED_L4T_VERSIONS`,
`SUPPORTED_CUDA_PREFIX`). JetPack 7.2.1 (Jetson Linux 39.2.1) also supports Orin but nobody has
tested this setup on it. Add it there only if you accept that risk. JetPack 6 needs a different
PyTorch build, which this script does not cover.

## Testing without a Jetson

On any other computer the script only does a dry run. To try the version checks, point it at
fake device files (this also forces a dry run):

```bash
SETUP_JETSON_TEST_UNAME="Linux aarch64" SETUP_JETSON_TEST_ROOT=/path/to/fake/root \
  jetson/setup_jetson.sh
```

The fake root needs `etc/nv_tegra_release`, `proc/device-tree/model`,
`usr/local/cuda/version.json` and `proc/meminfo`.
