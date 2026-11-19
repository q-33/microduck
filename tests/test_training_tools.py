"""Tests for duck_train/run_training.py, export_policy.py and replay_policy.py.

    uv run --project third_party/microduck_rl --with pytest pytest tests/test_training_tools.py -q -s
Skip the end-to-end run (train 16 envs x 2 iterations, export, replay; ~30 s on CPU)
with -m "not slow".
"""
from __future__ import annotations

import os
import re
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


_OWNED_VALUES = {"--gpu-ids": "None", "--env.scene.num-envs": "8", "--agent.max-iterations": "3",
                 "--agent.logger": "wandb", "--agent.resume": "True"}


def _spellings(flag, value):
    under = flag.replace("-", "_").replace("__", "--", 1)  # keep the leading "--"
    return [[flag, value], [under, value], [f"{flag}={value}"], [f"{under}={value}"]]


@pytest.mark.parametrize("bad", [s for f, v in _OWNED_VALUES.items() for s in _spellings(f, v)],
                         ids=lambda b: " ".join(b))
def test_owned_flags_refused_in_every_spelling(bad):
    assert set(_OWNED_VALUES) == set(rt.OWNED_FLAGS)
    with pytest.raises(ValueError, match="set by this launcher"):
        _cmd(extra=["--agent.save-interval", "5", *bad])


def test_extra_value_reads_any_spelling():
    assert rt.extra_value(["--agent.load_run", "a"], rt.LOAD_RUN) == "a"
    assert rt.extra_value(["--agent.load-run=b", "--agent.load-run", "c"], rt.LOAD_RUN) == "c"
    assert rt.extra_value(["--x", "1"], rt.LOAD_RUN) is None


@pytest.mark.parametrize("kw", [dict(device="gpu"), dict(envs=0), dict(iterations=0)])
def test_bad_values_refused(kw):
    with pytest.raises(ValueError):
        _cmd(**kw)


@pytest.mark.parametrize("name", ["", "../x", "a/b", "/abs", ".hidden", "smoke\n", "a b"])
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

    r = replay(onnx, seconds=2.0, num_envs=4, seed=3)
    print(f"[e2e replay] {r}")
    assert r["steps"] == 100 and r["input_shape"] == [1, rc.POLICY_OBS_SIZE]
    assert replay(onnx, seconds=2.0, num_envs=4, seed=3) == r  # same seed, same result


@pytest.mark.slow
def test_export_outside_repo_writes_no_absolute_path(tmp_path):
    """A checkpoint outside the repo: run_path metadata is parent/file, never absolute."""
    import onnxruntime as ort

    from duck_train.export_policy import newest_checkpoint

    name = "pytest_e2e"
    if not (ROOT / "out" / name).exists():
        pytest.skip("needs the end-to-end test's run (run the slow tests together)")
    assert not tmp_path.resolve().is_relative_to(ROOT)
    ckpt = tmp_path / "somerun" / "model_1.pt"
    ckpt.parent.mkdir()
    shutil.copy(newest_checkpoint(name), ckpt)
    r = subprocess.run([sys.executable, str(ROOT / "duck_train" / "export_policy.py"), "--checkpoint", str(ckpt)],
                       capture_output=True, text=True, timeout=900, cwd=tmp_path,
                       env={**os.environ, "MJLAB_WARP_QUIET": "1"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "run_path metadata written: somerun/model_1.pt" in r.stdout
    meta = ort.InferenceSession(str(ckpt.with_suffix(".onnx"))).get_modelmeta().custom_metadata_map
    home = str(Path.home())
    print(f"\n[outside export] run_path={meta['run_path']!r}")
    for key, value in meta.items():
        assert not value.startswith("/") and home not in value, (key, value)


# --------------------------------------------------------------------------- argument errors


@pytest.mark.parametrize("argv", [["--name", "smoke\n"], ["--name", "../x"], ["--name", "ok", "--envs", "0"],
                                  ["--name", "ok", "--iterations", "-1"], ["--name", "ok", "--envs", "x"],
                                  ["--name", "ok", "--env.scene.num_envs", "8"]])
def test_bad_arguments_exit_2_with_one_line(argv, capsys):
    with pytest.raises(SystemExit) as e:
        rt.main([*argv, "--device", "cpu", "--dry-run"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "error:" in err and "Traceback" not in err


def test_export_run_name_is_checked(capsys):
    from duck_train import export_policy

    with pytest.raises(SystemExit) as e:
        export_policy.main(["--run", "../smoke"])
    assert e.value.code == 2


def test_cuda_without_a_gpu_stops_before_launch():
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(SystemExit, match="torch sees no CUDA device"):
        rt.resolve_device("cuda")


def test_empty_cuda_visible_devices_warns(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert rt.resolve_device("cpu") == "cpu"
    assert "CUDA_VISIBLE_DEVICES is set but empty" in capsys.readouterr().err


# --------------------------------------------------------------------------- resume choice


def _run(exp: Path, name: str, *files: str) -> Path:
    d = exp / name
    (d / "params").mkdir(parents=True)
    for f in files:
        (d / f).write_bytes(b"")
    return d


def test_resume_skips_newer_folder_without_checkpoint(tmp_path):
    exp = tmp_path / "logs" / "rsl_rl" / rc.EXPERIMENT_NAME
    _run(exp, "2026-11-18_10-00-00_velocity_replica", "model_0.pt", "model_250.pt", "model_1000.pt")
    good = _run(exp, "2026-11-18_11-00-00_velocity_replica", "model_1000.pt", "model_1250.pt")
    _run(exp, "2026-11-18_12-00-00_velocity_replica")  # interrupted resume: no checkpoint
    (exp / "wandb_checkpoints").mkdir()
    load_run, run, ckpt = rt.plan_resume(tmp_path, rc.TASK_ID, [])
    assert run == good and ckpt == "model_1250.pt"
    assert re.match(load_run, good.name) and not re.match(load_run, good.name + "x")
    assert [n for n in (p.name for p in exp.iterdir()) if re.match(load_run, n)] == [good.name]
    cmd = _cmd(resume=True, load_run=load_run)
    assert cmd[-2:] == [rt.LOAD_RUN, load_run]


def test_resume_respects_user_load_run(tmp_path):
    exp = tmp_path / "logs" / "rsl_rl" / rc.EXPERIMENT_NAME
    old = _run(exp, "2026-11-18_10-00-00_velocity_replica", "model_5.pt")
    _run(exp, "2026-11-18_11-00-00_velocity_replica", "model_9.pt")
    load_run, run, ckpt = rt.plan_resume(tmp_path, rc.TASK_ID, ["--agent.load_run", ".*10-00"])
    assert load_run is None and run == old and ckpt == "model_5.pt"
    _run(exp, "2026-11-18_10-00-01_velocity_replica")
    with pytest.raises(SystemExit, match="has no checkpoint"):
        rt.plan_resume(tmp_path, rc.TASK_ID, ["--agent.load-run=.*10-00"])


def test_resume_with_other_task_refused(tmp_path):
    _run(tmp_path / "logs" / "rsl_rl" / rc.EXPERIMENT_NAME, "r1", "model_1.pt")
    with pytest.raises(SystemExit, match="pass the --task this run was trained with"):
        rt.plan_resume(tmp_path, rt.OFFICIAL_TASK, [])


# --------------------------------------------------------------------------- cleanup after a failed start


def test_failed_new_launch_removes_only_its_empty_folder(tmp_path):
    run_dir = tmp_path / "new"
    _run(run_dir / "logs" / "rsl_rl" / "e", "r1")
    assert rt.cleanup_failed_launch(run_dir, existed_before=False, runs_before=set()) == [run_dir]
    assert not run_dir.exists()
    _run(run_dir / "logs" / "rsl_rl" / "e", "r1", "model_0.pt")
    assert rt.cleanup_failed_launch(run_dir, existed_before=False, runs_before=set()) == []
    assert run_dir.exists()


def test_failed_resume_removes_only_new_checkpoint_free_runs(tmp_path):
    exp = tmp_path / "logs" / "rsl_rl" / "e"
    old_empty = _run(exp, "r0")  # existed before: kept even without a checkpoint
    old = _run(exp, "r1", "model_1.pt")
    before = set(tmp_path.glob("logs/rsl_rl/*/*"))
    new_empty = _run(exp, "r2")
    new_ckpt = _run(exp, "r3", "model_2.pt")
    assert rt.cleanup_failed_launch(tmp_path, existed_before=True, runs_before=before) == [new_empty]
    assert old_empty.exists() and old.exists() and new_ckpt.exists() and not new_empty.exists()


def test_failed_launch_end_to_end_leaves_no_folder():
    name = f"pytest_badflag_{os.getpid()}"
    r = subprocess.run([sys.executable, str(ROOT / "duck_train" / "run_training.py"), "--name", name,
                        "--device", "cpu", "--envs", "4", "--iterations", "1", "--no-such-upstream-flag", "1"],
                       capture_output=True, text=True, env={**os.environ, "MJLAB_WARP_QUIET": "1"}, timeout=600)
    assert r.returncode != 0
    assert "removed" in r.stderr and not (ROOT / "out" / name).exists()
