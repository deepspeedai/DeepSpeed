[![License MIT](https://badgen.net/badge/license/MIT/blue)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE)


<div align="center">
 <img src="docs/assets/images/DeepSpeed_light.svg#gh-light-mode-only" width="400px">
 <img src="docs/assets/images/DeepSpeed_dark_transparent.svg#gh-dark-mode-only" width="400px">
</div>

## DeeperSpeed

DeeperSpeed is a fork of the [Deepspeed](https://github.com/microsoft/DeepSpeed) library that is tailor-made for the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) by [EleutherAI](https://www.eleuther.ai/). 

Prior to 3/9/2023, DeeperSpeed was based on an old version of DeepSpeed (0.3.15). In order to migrate to the latest upstream DeepSpeed version while allowing users to access the old versions of GPT-NeoX and DeeperSpeed, we have introduced two versioned releases for both libraries:

- Version 1.0 of [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/releases/tag/v1.0) and [DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed/releases/tag/v1.0) maintain snapshots of the old stable versions that [GPT-NeoX-20B](https://arxiv.org/abs/2204.06745) and the [Pythia Suite](https://github.com/EleutherAI/pythia) were trained on.
- Version 2.0 of [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/releases/tag/v2.0) and [DeeperSpeed](https://github.com/EleutherAI/DeeperSpeed/releases/tag/v2.0) are the latest versions built on the latest DeepSpeed, and will be maintained going forward.
