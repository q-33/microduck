"""Tests for duck_train/run_training.py, export_policy.py and replay_policy.py.

    uv run --project third_party/microduck_rl --with pytest pytest tests/test_training_tools.py -q -s
Skip the end-to-end run (train 16 envs x 2 iterations, export, replay; ~30 s on CPU)
with -m "not slow".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from duck_train import replica_config as rc
from duck_train import run_training as rt

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- launcher mapping


def _cmd(**kw):
    base = dict(task=rt.REPLICA_TASK, envs=64, iterations=5, device="cpu", resume=False, python="py")
    return rt.build_command(**{**base, **kw})


def test_cpu_maps_to_gpu_ids_none():
    cmd = _cmd()
    assert cmd[:3] == ["py", str(ROOT / "duck_train" / "train.py"), rc.TASK_ID]
    assert cmd[3:] == ["--gpu-ids", "None", "--env.scene.num-envs", "64",
                       "--agent.max-iterations", "5", "--agent.logger", "tensorboard"]


def test_cuda_maps_to_first_gpu():
    cmd = _cmd(device="cuda", envs=512, iterations=1000)
    assert cmd[cmd.index("--gpu-ids") + 1] == "[0]"
    assert cmd[cmd.index("--env.scene.num-envs") + 1] == "512"
    assert cmd[cmd.index("--agent.max-iterations") + 1] == "1000"


def test_resume_and_extras_passed_through_unchanged():
    extra = ["--agent.save-interval", "100", "--agent.load-run", ".*_velocity_replica"]
    cmd = _cmd(resume=True, extra=extra, task=rt.OFFICIAL_TASK)
    assert cmd[2] == rt.OFFICIAL_TASK
    assert cmd[-len(extra):] == extra
    assert cmd[cmd.index("--agent.resume") + 1] == "True"
    assert "--agent.resume" not in _cmd()


@pytest.mark.parametrize("bad", [["--gpu-ids", "[1]"], ["--env.scene.num-envs=8"],
                                 ["--agent.max_iterations", "3"], ["--agent.resume", "True"]])
def test_owned_flags_refused_as_extras(bad):
    with pytest.raises(ValueError):
        _cmd(extra=bad)


@pytest.mark.parametrize("kw", [dict(device="gpu"), dict(envs=0), dict(iterations=0)])
def test_bad_values_refused(kw):
    with pytest.raises(ValueError):
        _cmd(**kw)


@pytest.mark.parametrize("name", ["", "../x", "a/b", "/abs", ".hidden"])
def test_run_name_must_be_a_plain_folder_name(name):
    with pytest.raises(ValueError):
        rt.run_dir_for(name)


def test_run_dir_is_under_repo_out():
    assert rt.run_dir_for("smoke") == ROOT / "out" / "smoke"


def test_existing_run_dir_refused_unless_resume(tmp_path):
    rt.check_run_dir(tmp_path / "new", resume=False)  # fine
    with pytest.raises(SystemExit, match="already exists"):
        rt.check_run_dir(tmp_path, resume=False)
    with pytest.raises(SystemExit, match="no checkpoint"):
        rt.check_run_dir(tmp_path, resume=True)
    ckpt = tmp_path / "logs" / "rsl_rl" / "velocity_replica" / "2026_run" / "model_4.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    rt.check_run_dir(tmp_path, resume=True)  # fine


def test_parse_keeps_unknown_args_for_upstream():
    args, extra = rt.parse(["--envs", "1024", "--name", "orin", "--agent.save-interval", "100"])
    assert args.envs == 1024 and args.name == "orin" and args.device == "auto"
    assert extra == ["--agent.save-interval", "100"]


@pytest.mark.parametrize("device,gpu_ids", [("cpu", None), ("cuda", [0])])
def test_upstream_cli_accepts_the_mapped_flags(device, gpu_ids):
    """Parse our flags with upstream's real TrainConfig and tyro settings (no training)."""
    import tyro

    import duck_train.replica_task  # noqa: F401  (registers the replica task)
    import mjlab
    from mjlab.scripts.train import TrainConfig

    cmd = _cmd(device=device, envs=48, iterations=7, resume=True, extra=["--agent.save-interval", "3"])
    cfg = tyro.cli(TrainConfig, args=cmd[3:], default=TrainConfig.from_task(rc.TASK_ID),
                   config=mjlab.TYRO_FLAGS)
    assert cfg.gpu_ids == gpu_ids
    assert cfg.env.scene.num_envs == 48
    assert cfg.agent.max_iterations == 7
    assert cfg.agent.logger == "tensorboard"
    assert cfg.agent.resume is True
    assert cfg.agent.save_interval == 3


def test_dry_run_prints_command_from_any_folder(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    name = f"pytest_dry_{os.getpid()}"
    assert rt.main(["--name", name, "--device", "cpu", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"working folder: {ROOT / 'out' / name}" in out
    assert "--gpu-ids None --env.scene.num-envs 64 --agent.max-iterations 5" in out
    assert not (ROOT / "out" / name).exists()


# --------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_train_export_replay_end_to_end():
    import numpy as np
    import onnxruntime as ort

    from duck_train.export_policy import check_onnx, newest_checkpoint
    from duck_train.replay_policy import replay

    name = "pytest_e2e"
    run_dir = ROOT / "out" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    env = {**os.environ, "MJLAB_WARP_QUIET": "1"}
    py = sys.executable

    def run(args):
        r = subprocess.run([py, *args], capture_output=True, text=True, timeout=900, env=env, cwd="/")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        return r.stdout

    out = run([str(ROOT / "duck_train" / "run_training.py"), "--envs", "16", "--iterations", "2",
               "--device", "cpu", "--name", name])
    assert "--gpu-ids None --env.scene.num-envs 16 --agent.max-iterations 2" in out
    ckpt = newest_checkpoint(name)
    assert ckpt.name == "model_1.pt" and ckpt.is_relative_to(run_dir / "logs" / "rsl_rl" / rc.EXPERIMENT_NAME)

    # Refuses to start over an existing run.
    r = subprocess.run([py, str(ROOT / "duck_train" / "run_training.py"), "--name", name, "--device", "cpu"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "already exists" in r.stderr

    out = run([str(ROOT / "duck_train" / "export_policy.py"), "--run", name])
    assert "[export_policy] OK" in out and "FAIL" not in out
    onnx = ckpt.with_suffix(".onnx")
    print("\n" + "\n".join(check_onnx(onnx, rc.TASK_ID)))

    # Same final weights as the ONNX upstream wrote at the end of training.
    train_onnx = next(ckpt.parent.glob(f"{ckpt.parent.name}.onnx"))
    x = np.random.default_rng(0).normal(size=(1, rc.POLICY_OBS_SIZE)).astype(np.float32)
    ys = [s.run(None, {s.get_inputs()[0].name: x})[0]
          for s in (ort.InferenceSession(str(p)) for p in (onnx, train_onnx))]
    assert np.allclose(ys[0], ys[1], atol=1e-6)

    r = replay(onnx, seconds=2.0)
    print(f"[e2e replay] {r}")
    assert r["steps"] == 100 and r["input_shape"] == [1, rc.POLICY_OBS_SIZE]
