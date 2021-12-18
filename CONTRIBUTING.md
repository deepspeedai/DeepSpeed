# Contributing
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
[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) and measure
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
To execute model tests, first [install DeepSpeed](#installation). The
[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) repository is cloned
as part of this process. Next, execute the model test driver:
```bash
cd tests/model/
pytest run_sanity_check.py
```
Note that the `--forked` flag is not necessary for the model tests.

## Contributor License Agreement
This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. For details, visit
https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across
all repos using our CLA.

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.

## New Feature Contribution Guidelines
Unlike bug fix or improving existing feature (where users usually directly submit a PR and we review it), adding a new feature to DeepSpeed requires several steps: (1) proposal and discussion, (2) implementation and verification, (3) release and maintenance. This general guideline applies to all new feature contributions. Core DeepSpeed team member contributions may complete step 1 internally.

### Step 1: proposal and discussion
We ask users to first post your intended feature in an issue. This issue needs to include:

* A description of the proposed feature.
* A motivation of why it will be useful to DeepSpeed users.
* A rough design of how you implement the feature inside DeepSpeed.
* (Important) Results or planned experiments to demonstrate the effectiveness and correctness of the feature.
  * If this is a general feature applicable to different tasks, we require testing it on at least one CV task (e.g., [CIFAR](https://www.deepspeed.ai/tutorials/cifar-10/)) and one NLP task (e.g., [SQuAD](https://www.deepspeed.ai/tutorials/bert-finetuning/)). If this is a feature for one kind of task only, it is fine to just test on the specific task.
  * If the feature only affects performance and does not affect training convergence, we require testing on a fraction of training to demonstrate that the training/validation loss are consistent with baseline, and that the performance is better than baseline.
  * If the feature does affect training convergence, we require testing the whole training to demonstrate that the feature achieves better/on-par final model quality and training performance compared to baseline.

Based on the issue we shall discuss the merit of the new feature and decide whether accept or decline the proposal. Once accepted and after we confirm the design and implementation plan, we are ready for step 2.

### Step 2: implementation and verification
Contributor will go ahead and implement the feature, and the DeepSpeed team will provide guidance/helps as needed. The required deliverables include:

* A PR to [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) including (1) the feature implementation (2) unit tests (3) documentation (4) tutorial
* A PR to [microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) or [microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) including the examples of how to use the feature (this is related to the planned testing experiments in proposal)
* In the implementation (code, documentation, tutorial), we require the feature author to record their GitHub username as a contact method for future questions/maintenance.

After receiving the PRs, we will review them and merge them after necessary tests/fixes.

### Step 3: release and maintenance
After the PRs are merged, we will announce the feature on our website (with credit to the feature author). We ask the feature author to commit to the maintenance of the feature.
