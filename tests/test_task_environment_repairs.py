import json
import subprocess
import sys
import threading

from tact.config import resolve_config
from tact.environments.containers import ContainerFactory
from tact.environments.policy import EnvironmentPolicy, PROBE_CODE


def test_task_repair_isolated_and_shared_by_container_spec(tmp_path):
    cfg = resolve_config()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "repair.whl").write_bytes(b"repair wheel identity")
    factory = ContainerFactory.__new__(ContainerFactory)
    factory.cfg = cfg
    factory.policy = EnvironmentPolicy(cfg)
    factory._task_policies = {}
    factory._lock = threading.Lock()
    factory.images = {"repaired": {"image_id": "sha256:fixed"}}
    factory.lock_images = lambda rows: factory.images
    original = {"instance_id": "original", "repo": "sympy/sympy"}
    repaired = {"instance_id": "repaired", "repo": "sympy/sympy",
                "environment_overrides": {"wheelhouse": str(wheels),
                    "repo_pins": {"sympy/sympy": ["scipy==1.13.1"]},
                    "variables": {"OMP_NUM_THREADS": "1"}}}
    assert factory.policy_for(original) is factory.policy
    policy = factory.policy_for(repaired)
    assert "scipy==1.13.1" in policy.bootstrap(repaired["repo"])
    spec = factory.spec(repaired)
    assert spec["environment"]["OMP_NUM_THREADS"] == "1"
    assert str(wheels) in spec["volumes"]
    assert spec["policy_hash"] == policy.identity(repaired["repo"])["policy_hash"]
    assert "OMP_NUM_THREADS" not in factory.policy.variables()
    assert "scipy==1.13.1" not in factory.policy.specs(original["repo"])


def test_probe_decodes_pip_json_before_old_python_json_loads():
    # Python 3.5 rejects json.loads(bytes); emulate that contract on the host.
    prelude = """import json,subprocess
real_loads=json.loads
def str_only(value):
    assert isinstance(value,str), 'Python 3.5 json input must be text'
    return real_loads(value)
json.loads=str_only
subprocess.check_output=lambda *a,**k:b'[]'
import builtins,io
real_open=builtins.open
builtins.open=lambda name,*a,**k:io.StringIO('') if name=='/tmp/tact-constraints.txt' else real_open(name,*a,**k)
"""
    result = subprocess.run([sys.executable, "-c", prelude + PROBE_CODE],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["constraints"] == []
