# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Install Accelerate's test extras without replacing the CI framework stack."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ACCELERATE_REPOSITORY = "https://github.com/huggingface/accelerate.git"
ACCELERATE_REF = "8c83b448d19f6356d4facd65c1d4763bf9fa51b2"
PROTECTED_PACKAGES = ("torch", "torchvision", "torchaudio", "datasets")
REPORTED_PACKAGES = PROTECTED_PACKAGES + ("accelerate", "transformers", "pytest")


def package_versions(packages):
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run(command, **kwargs):
    print("running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def install(destination: Path, ref: str):
    before = package_versions(PROTECTED_PACKAGES)
    missing = [package for package, version in before.items() if version is None]
    if missing:
        raise RuntimeError(f"protected packages must be installed first: {', '.join(missing)}")

    run(("git", "clone", "--filter=blob:none", ACCELERATE_REPOSITORY, str(destination)))
    run(("git", "checkout", "--detach", ref), cwd=destination)
    resolved_ref = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=destination, text=True).strip()
    if resolved_ref != ref:
        raise RuntimeError(f"Accelerate resolved to {resolved_ref}, expected {ref}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt") as constraints:
        for package, version in before.items():
            constraints.write(f"{package}=={version}\n")
        constraints.flush()
        run((sys.executable, "-m", "pip", "install", "-c", constraints.name, f"{destination}[testing]"))

    after = package_versions(PROTECTED_PACKAGES)
    if after != before:
        raise RuntimeError(f"Accelerate installation changed protected packages: before={before}, after={after}")

    run((sys.executable, "-m", "pip", "check"))
    print(f"Accelerate resolved ref: {resolved_ref}")
    print("Environment:", json.dumps(package_versions(REPORTED_PACKAGES), sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ref", default=ACCELERATE_REF)
    args = parser.parse_args()
    install(args.destination.resolve(), args.ref)


if __name__ == "__main__":
    main()
