"""Train a task, including our replica task, with upstream's unchanged `train` command.

mjlab finds tasks through the "mjlab.tasks" entry-point group of INSTALLED packages
(mjlab/__init__.py: _import_registered_packages). duck_train is not installed in
upstream's environment, so upstream's `train` cannot see our task. This wrapper
registers it, then hands every argument to upstream's own train entry point
(mjlab_microduck.train_cli:main -> mjlab.scripts.train:main).

Logs go to logs/ under the CURRENT folder, so run from a folder under out/:
    mkdir -p out/replica_smoke && cd out/replica_smoke
    uv run --project ../../third_party/microduck_rl python ../../duck_train/train.py \
        Mjlab-Velocity-Flat-MicroDuck-Replica --gpu-ids None \
        --env.scene.num-envs 16 --agent.max-iterations 2 --agent.logger tensorboard
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import duck_train.replica_task  # noqa: F401  (registers the replica task)
    from mjlab_microduck.train_cli import main as upstream_train_main

    return upstream_train_main()


if __name__ == "__main__":
    sys.exit(main())
