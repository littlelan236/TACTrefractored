from __future__ import annotations

import json
import shlex
from pathlib import Path

from tact.artifacts import hash_json, sha256

CONSTRAINT_FILE = "/tmp/tact-constraints.txt"
WHEELS_PATH = "/opt/tact-wheels"


class EnvironmentPolicy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.environment = cfg["environment"]
        wheelhouse = Path(self.environment["wheelhouse"])
        self.wheel_hashes = {p.name: sha256(p) for p in sorted(wheelhouse.glob("*.whl"))}
        if not self.wheel_hashes:
            raise ValueError("Wheelhouse is empty")

    def specs(self, repo: str) -> list[str]:
        return list(dict.fromkeys(self.environment["constraints"] +
                                  self.environment["repo_pins"].get(repo, [])))

    def variables(self) -> dict[str, str]:
        env = dict(self.environment["variables"])
        env.update({"PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
                    "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"})
        env.update({"LANG": self.environment["locale"], "LC_ALL": self.environment["locale"],
                    "PYTHONIOENCODING": "utf-8", "PIP_CONSTRAINT": CONSTRAINT_FILE,
                    "PIP_FIND_LINKS": WHEELS_PATH, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        return env

    def bootstrap(self, repo: str) -> str:
        specs = self.specs(repo)
        text = shlex.quote("\n".join(specs) + "\n")
        install = "python -m pip install --no-input --no-index --find-links " + WHEELS_PATH
        install += " " + " ".join(map(shlex.quote, specs))
        if self.environment["network_fallback"]:
            install = "(" + install + ") || (echo TACT_NETWORK_FALLBACK; python -m pip install "
            install += "--no-input " + " ".join(map(shlex.quote, specs)) + ")"
        return "\n".join([
            "set -euo pipefail", "source /opt/miniconda3/bin/activate", "conda activate testbed",
            f"printf '%s' {text} > {CONSTRAINT_FILE}",
            *(f"export {k}={shlex.quote(v)}" for k, v in self.variables().items()),
            "echo TACT_BOOTSTRAP_BEGIN", install, "echo TACT_BOOTSTRAP_OK",
        ]) + "\n"

    def identity(self, repo: str) -> dict:
        value = {"environment": self.environment, "timeouts": self.cfg["timeouts"],
                 "repo": repo, "specs": self.specs(repo), "wheel_hashes": self.wheel_hashes,
                 "bootstrap": self.bootstrap(repo)}
        return {"policy_hash": hash_json(value), **value}

    def wrap_eval(self, script: str) -> str:
        # Export before the official conda activation and every subsequent pip install.
        exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in self.variables().items())
        head, separator, rest = script.partition("\n")
        return head + separator + exports + "\n" + rest


PROBE_CODE = """import json,sys,subprocess,importlib,os,re
from importlib.metadata import version,PackageNotFoundError
try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
constraints=open('/tmp/tact-constraints.txt').read().splitlines()
checks=[]
for text in constraints:
    if not text.strip():continue
    requirement=Requirement(text)
    name=requirement.name
    try: actual=version(name)
    except PackageNotFoundError: actual=None
    checks.append({'spec':text,'package':name,'actual':actual,'satisfied':actual is not None and actual in requirement.specifier})
data={'python':sys.version,'executable':sys.executable,'conda':os.environ.get('CONDA_DEFAULT_ENV'),
      'constraints':checks,'packages':json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'])),
      'variables':{k:os.environ.get(k) for k in ['LANG','LC_ALL','PIP_CONSTRAINT','PIP_FIND_LINKS']}}
print(json.dumps(data))
sys.exit(0 if all(c['satisfied'] for c in checks) else 3)
"""


def probe_command() -> str:
    return "source /opt/miniconda3/bin/activate && conda activate testbed && python -c " + shlex.quote(PROBE_CODE)


def parse_probe(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "constraints" in value:
            return value
    raise ValueError("Environment probe did not return JSON")
