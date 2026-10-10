from tact.backends.native_vllm import model_config
from tact.config import resolve_config
from tact.environments.policy import CONSTRAINT_FILE, EnvironmentPolicy


def test_repo_specific_pins_apply_to_future_installs_and_eval():
    policy = EnvironmentPolicy(resolve_config())
    text = policy.bootstrap("pydicom/pydicom")
    assert "numpy<1.24" in text and "pytest<8" in text
    assert "set -euo pipefail" in text
    assert "--no-index" in text
    assert " || " not in text
    variables = policy.variables()
    assert variables["PIP_CONSTRAINT"] == CONSTRAINT_FILE
    official = "#!/bin/bash\nconda activate testbed\npip install -e .\npytest\n"
    script = policy.wrap_eval(official)
    assert official.splitlines()[1:] == script.splitlines()[-3:]
    assert script.index("export PIP_CONSTRAINT=") < script.index("pip install -e .")


def test_baseline_generation_has_no_silent_drop_or_steering_fields():
    cfg = resolve_config(overrides=["generation.seed=23"])
    kwargs = model_config(cfg)["model_kwargs"]
    assert kwargs["seed"] == 23
    assert "drop_params" not in kwargs
    assert "tact" not in kwargs["extra_body"] and "steering" not in kwargs["extra_body"]
