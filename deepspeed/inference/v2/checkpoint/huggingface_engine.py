# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import json
import torch
from .base_engine import CheckpointEngineBase
from typing import Iterable, Tuple
from functools import partial

from ..logging import inference_logger


class HuggingFaceCheckpointEngine(CheckpointEngineBase):

    def __init__(self, model_name_or_path: str, auth_token: str = None) -> None:
        super().__init__()
        from transformers import AutoConfig, GenerationConfig

        self.model_name_or_path = model_name_or_path
        self.auth_token = auth_token
        self.model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        # Define this property here so we can use it in the model implementation
        if not hasattr(self.model_config, "max_seq_length"):
            if hasattr(self.model_config, "max_position_embeddings"):
                self.model_config.max_seq_length = self.model_config.max_position_embeddings
            else:
                generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
                self.model_config.max_seq_length = generation_config.max_length
        self._local_checkpoint_dir = None
        self._all_ckpt_paths = self._fetch_checkpoint_files()

    def _fetch_checkpoint_files(self):
        """
        Fetch the checkpoint files from the HuggingFace Hub.
        """
        # TODO(jeff): for models like llama-2 the user will have to provide an auth `token`,
        # currently coming from the ckpt engine init but maybe a catch all kwargs for other
        # snapshot download parameters would be more flexible.

        from huggingface_hub import snapshot_download, list_repo_tree

        def model_has_safetensors(model_name_or_path: str) -> bool:
            if os.path.isdir(model_name_or_path):
                file_list = os.listdir(model_name_or_path)
            else:
                file_list = [rf.path for rf in list_repo_tree(model_name_or_path)]
            for f in file_list:
                if f.endswith(".safetensors"):
                    return True
            return False

        if os.path.isdir(self.model_name_or_path):
            self._local_checkpoint_dir = self.model_name_or_path
        else:
            # We need to download the checkpoint files from HF
            if model_has_safetensors(self.model_name_or_path):
                # Prioritize downloading safetensors if they are available
                allow_patterns = ["*.safetensors", "*.json"]
            else:
                # Fallback to bin files when safetensors are not present
                allow_patterns = ["*.bin", "*.json", "*.pt"]
            self._local_checkpoint_dir = snapshot_download(self.model_name_or_path,
                                                           allow_patterns=allow_patterns,
                                                           revision=None,
                                                           token=self.auth_token)

        assert os.path.isdir(
            self._local_checkpoint_dir
        ), f"Checkpoint dir {self._local_checkpoint_dir} is not a directory, cannot load checkpoint."

        # Set the appropriate file names based on whether we have safetensors or not
        if model_has_safetensors(self._local_checkpoint_dir):
            from safetensors.torch import load_file
            model_param_json_fname = "model.safetensors.index.json"
            model_file_fname = "model.safetensors"
            self._checkpoint_load_fn = load_file
        else:
            model_param_json_fname = "pytorch_model.bin.index.json"
            model_file_fname = "pytorch_model.bin"
            self._checkpoint_load_fn = partial(torch.load, map_location="cpu")

        model_param_json = os.path.join(self._local_checkpoint_dir, model_param_json_fname)

        if not os.path.isfile(model_param_json):
            # We don't need any json as all such HF models will have pytorch_model.bin
            all_checkpoint_files = [os.path.join(self._local_checkpoint_dir, model_file_fname)]
        else:
            param_map = json.load(open(model_param_json, "r"))

            # weight_map -> { "lm_head.weight": "pytorch_model-00002-of-00002.bin", ... }
            weight_map = param_map["weight_map"]

            # unique set of all checkpoint files
            all_checkpoint_files = set(weight_map.values())

            # get absolute path of all unique checkpoint files
            all_checkpoint_files = [os.path.join(self._local_checkpoint_dir, f) for f in all_checkpoint_files]

        return all_checkpoint_files

    def parameters(self) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Generator of model parameters (satisfies the CheckpointEngineBase interface).
        """
        for checkpoint in self._all_ckpt_paths:
            inference_logger().info(f"Loading checkpoint: {checkpoint}")
            checkpoint_sd = self._checkpoint_load_fn(checkpoint)
            param_keys = list(checkpoint_sd.keys())
            for param_name in param_keys:
                param = checkpoint_sd[param_name]
                yield param_name, param

            del checkpoint_sd


if __name__ == "__main__":
    # To test, add your auth_token here and run `python huggingface_engine.py`
    engine = HuggingFaceCheckpointEngine(model_name_or_path="meta-llama/Llama-2-7b-hf",
                                         auth_token="hf_xxxxxxxxxxxxxxxxx")
    for name, param in engine.parameters():
        print(name, param.shape)
