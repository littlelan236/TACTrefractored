from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

from tact.rollout import runner


def test_staged_rollout_waits_for_axes_before_eval(monkeypatch, tmp_path):
    """One slow training task must not let val/eval start in another worker."""
    train_started, val_started, eval_started = Event(), Event(), Event()
    release_train, release_val = Event(), Event()
    rows = [{"instance_id": iid} for iid in ("train-fast", "train-slow", "val", "eval")]
    partitions = {"axis-train": ["train-fast", "train-slow"], "axis-val": ["val"], "eval": ["eval"]}
    monkeypatch.setattr(runner, "select_instances", lambda cfg: (rows, partitions))
    factory = Mock()
    factory.lock_images.return_value = {}
    monkeypatch.setattr(runner, "ContainerFactory", lambda *args: factory)
    backend = Mock()
    backend.__enter__ = Mock(return_value=backend)
    backend.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(runner, "process_manager", lambda *args: backend)
    monkeypatch.setattr(runner, "generate_predictions", lambda *args: [])

    def task(cfg, run, row, factory):
        iid = row["instance_id"]
        if iid == "train-slow":
            train_started.set()
            assert release_train.wait(5)
        elif iid == "val":
            val_started.set()
            assert release_val.wait(5)
        elif iid == "eval":
            eval_started.set()
        return {"instance_id": iid, "exit_status": "Submitted"}

    monkeypatch.setattr(runner, "process_task", task)
    cfg = {"execution": {"rollout_workers": 2, "staged_partitions": True}}
    with ThreadPoolExecutor(max_workers=1) as control:
        result = control.submit(runner.run_rollout, cfg, tmp_path)
        try:
            assert train_started.wait(2)
            assert not val_started.wait(0.1)
            assert not eval_started.is_set()
            release_train.set()
            assert val_started.wait(2)
            assert not eval_started.wait(0.1)
            release_val.set()
            assert result.result(timeout=5)["complete"]
            assert eval_started.is_set()
        finally:
            release_train.set()
            release_val.set()
