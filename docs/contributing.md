---
title: "Contributing"
permalink: /contributing/
---

DeepSpeed welcomes your contributions!

## Prerequisites
DeepSpeed uses [pre-commit](https://pre-commit.com/) to ensure that formatting is
consistent across DeepSpeed. First, ensure that `pre-commit` is installed from either
installing DeepSpeed or `pip install pre-commit`. Next, the pre-commit hooks must be
installed once before commits can be made:
```bash
pre-commit install
```

Afterwards, our suite of formatting tests run automatically before each `git commit`. You
can also run these manually:
```bash
pre-commit run --all-files
```
If a formatting test fails, it will fix the modified code in place and abort
the `git commit`. After looking over the changes, you can `git add <modified files>`
and then repeat the previous `git commit` command.


## Testing
DeepSpeed tracks two types of tests: unit tests and more costly model convergence tests.
The model convergence tests train
[DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/) and measure
end-to-end convergence and related metrics. Unit tests are found in `tests/unit/` and
the model convergence tests are found in `tests/model/`.

### Unit Tests
[PyTest](https://docs.pytest.org/en/latest/) is used to execute tests. PyTest can be
installed from PyPI via `pip install pytest`. Simply invoke `pytest --forked` to run the
unit tests:
```bash
pytest --forked tests/unit/
```
You can also provide the `-v` flag to `pytest` to see additional information about the
tests. Note that [pytest-forked](https://github.com/pytest-dev/pytest-forked) and the
`--forked` flag are required to test CUDA functionality in distributed tests.

### Model Tests
Model tests require four GPUs and training data downloaded for
[DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/).

To execute model tests, first [install DeepSpeed](/getting-started/#installation). The
[DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/) repository is cloned
as part of this process. Next, execute the model test driver:
```bash
cd tests/model/
pytest run_sanity_check.py
```
Note that the `--forked` flag is not necessary for the model tests.

## Contributor License Agreement
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Developer Certificate of Origin (DCO)[https://wiki.linuxfoundation.org/dco]
stating that they agree to the terms published at https://developercertificate.org for
that *particular* contribution.

DCOs are per-commit, so each commit needs to be signed off.  These can be signed in
the commit by adding the `-s` flag.  DCO enforcement can also be signed off in the PR
itself by clicking on the DCO enforcement check.

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.
