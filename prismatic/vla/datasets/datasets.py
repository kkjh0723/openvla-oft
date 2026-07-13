"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights

@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_temporal_image: bool = False  # If True, also emit the OLDEST primary frame in the obs window (t-2) as an extra image
    use_delta_image: bool = False     # If True, emit the materialized ΔI channel (image_delta_primary, current step) as the extra image INSTEAD of the raw t-2 frame (OFT-ΔI). Mutually exclusive with use_temporal_image.

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # The obs window axis (index 0 of each observation tensor) holds
        # [t-(W-1), ..., t-1, t]; the CURRENT frame is the LAST element (index -1),
        # index 0 is the oldest. chunk_act_obs aligns the action axis the same way:
        # action = [a_{t-(W-1)}, ..., a_{t-1}, a_t, a_{t+1}, ..., a_{t+future}], so
        # the current action lives at index (W-1), NOT 0. We derive W from the
        # primary-image window length so this is correct for any window_size.
        # For window_size==1 this reduces to the original [-1]==[0]==t behavior.
        window_size = int(rlds_batch["observation"]["image_primary"].shape[0])
        cur_idx = window_size - 1  # index of the current step on the windowed axes
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][cur_idx]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # `actions` returned to the collator must be the chunk [t : t+future]
        # (current + future) — drop the (W-1) PAST actions added by the obs window.
        actions = rlds_batch["action"][cur_idx:]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk (everything strictly after the current step)
        future_actions = rlds_batch["action"][cur_idx + 1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    # Use the CURRENT frame ([-1]) of the obs window, not the oldest.
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][-1])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        # Third image slot (emitted as pixel_values_temporal, concatenated LAST by
        # the collator so the model image order is [primary_t, wrist_t, extra]):
        #  - use_delta_image (OFT-ΔI): the materialized ΔI channel at the CURRENT
        #    step (image_delta_primary[-1]). ΔI is a per-step difference, so no
        #    history window is needed (window_size can be 1). Takes precedence.
        #  - use_temporal_image (OFT-2frame): the OLDEST window frame (image_primary
        #    [0] == t-2 with window_size=3 == GR00T d2 delta -2).
        # The two are mutually exclusive (ΔI REPLACES the raw t-2 frame). The eval
        # server attaches the analogous frame under obs["temporal_image"].
        if self.use_delta_image:
            img_delta = Image.fromarray(rlds_batch["observation"]["image_delta_primary"][-1])
            return_dict["pixel_values_temporal"] = self.image_transform(img_delta)
        elif self.use_temporal_image:
            img_temporal = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
            return_dict["pixel_values_temporal"] = self.image_transform(img_temporal)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            # CURRENT-frame proprio only (index cur_idx == -1). With window_size=3
            # the RLDS obs window stacks proprio as (W, proprio_dim); passing the
            # whole window makes the collator emit (bsz, W*proprio_dim) and the
            # 8-dim proprio_projector fails ("mat1 2x24 and mat2 8x4096"). 2-frame
            # is a VISUAL motion cue only — proprio history is not used. For
            # window_size==1 this reduces to the original single-frame behavior.
            proprio = rlds_batch["observation"]["proprio"][cur_idx]
            return_dict["proprio"] = proprio

        return return_dict


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 1,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders.

        ``window_size`` controls the observation history window. The default (1)
        keeps the original single-frame behavior. ``window_size=3`` stacks
        [t-2, t-1, t] on the obs window axis so RLDSBatchTransform can emit the
        t-2 primary frame for 2-frame (temporal) training.
        """
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")
        # OFT-ΔI: also load the materialized ΔI base channel (registered as
        # "delta_primary" in the OXE config). RLDSBatchTransform emits it as the
        # 3rd image in place of the raw t-2 frame.
        if getattr(self.batch_transform, "use_delta_image", False):
            load_camera_views = load_camera_views + ("delta_primary",)

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # >1 stacks past obs (t-(W-1)..t) for temporal input
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            _full_aug = dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )
            if getattr(self.batch_transform, "use_delta_image", False):
                # OFT-ΔI: per-image-name mapping (no top-level augment_order → the
                # mapping form, see obs_transforms.augment). Raw views keep FULL
                # aug; the ΔI view (delta_primary) is CROP-ONLY. Its crop is made
                # identical to primary's by the base-view seed-sharing in
                # obs_transforms.augment, so ΔI stays spatially aligned with the
                # current frame while its zero-motion anchor / magnitude are left
                # untouched by photometric jitter.
                _crop_only = dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    augment_order=["random_resized_crop"],
                )
                image_augment_kwargs = {
                    "primary": _full_aug,
                    "wrist": _full_aug,
                    "delta_primary": _crop_only,
                }
            else:
                image_augment_kwargs = _full_aug
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": image_augment_kwargs})
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
