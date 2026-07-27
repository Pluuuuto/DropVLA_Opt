"""Fine-tune OpenVLA via LoRA and save only the final checkpoint."""

import os
import time
import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import draccus
import torch
import torch.nn as nn
import tqdm
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import PartialState

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_DISABLE_DTENSOR"] = "1"
os.environ["TRANSFORMERS_NO_TENSOR_PARALLEL"] = "1"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 4                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

    # Poison-aware loss (disabled when weight == 0)
    poison_trigger_text: str = "carefully"           # Text marker used to identify a poisoned current step
    poison_gripper_loss_weight: float = 0.0           # Auxiliary L1 weight on poisoned current-step gripper action
    console_log_freq: int = 100                       # Print training-quality metrics every N optimizer steps

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # Reproducibility
    seed: Optional[int] = None                       # Random seed for reproducibility (if None, no seeding is applied)

    # fmt: on


@dataclass
class PoisonAwareRLDSBatchTransform:
    """Add minimal poison metadata without changing the underlying RLDS dataset.

    The current DropVLA_Opt RLDS schema does not contain an explicit per-step poison
    flag. For the joint ``carefully`` dataset, the textual trigger is written only
    on the poisoned current step, so the current action (offset 0) can be marked
    reliably. Future offsets are intentionally left unmarked rather than inferred
    from normal/open gripper labels.
    """

    base_transform: RLDSBatchTransform
    trigger_text: str = "carefully"

    @staticmethod
    def _decode_language(value: Any) -> str:
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = value.reshape(-1)[0]
            else:
                value = value.tolist()
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        output = self.base_transform(rlds_batch)

        language = self._decode_language(rlds_batch["task"]["language_instruction"]).lower()
        trigger = self.trigger_text.strip().lower()
        is_poison = bool(trigger) and trigger in language

        poison_action_mask = np.zeros((NUM_ACTIONS_CHUNK,), dtype=np.bool_)
        if is_poison:
            poison_action_mask[0] = True

        output["is_poison"] = np.bool_(is_poison)
        output["poison_action_mask"] = poison_action_mask
        output["trigger_offset"] = np.int64(0 if is_poison else -1)

        return output


@dataclass
class PoisonAwareCollator:
    """Preserve poison metadata that the stock OpenVLA-OFT collator drops."""

    base_collator: PaddedCollatorForActionPrediction

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        output = self.base_collator(instances)
        output["is_poison"] = torch.tensor(
            [bool(instance["is_poison"]) for instance in instances],
            dtype=torch.bool,
        )
        output["poison_action_mask"] = torch.stack(
            [
                torch.as_tensor(instance["poison_action_mask"], dtype=torch.bool)
                for instance in instances
            ],
            dim=0,
        )
        output["trigger_offset"] = torch.tensor(
            [int(instance["trigger_offset"]) for instance in instances],
            dtype=torch.long,
        )
        return output


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.seed is not None:
            run_id += f"--seed{cfg.seed}"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a component checkpoint from a checkpoint directory.

    New checkpoints always use the fixed filename
    ``<module_name>--latest_checkpoint.pt`` because the optimization step is
    already encoded in the directory name. For compatibility, legacy
    ``<module_name>--<step>_checkpoint.pt`` files are still accepted.
    """
    checkpoint_dir = Path(path)
    latest_path = checkpoint_dir / f"{module_name}--latest_checkpoint.pt"
    legacy_path = checkpoint_dir / f"{module_name}--{step}_checkpoint.pt"

    if latest_path.is_file():
        checkpoint_path = latest_path
    elif legacy_path.is_file():
        checkpoint_path = legacy_path
    else:
        candidates = sorted(checkpoint_dir.glob(f"{module_name}--*_checkpoint.pt"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one checkpoint for {module_name} in {checkpoint_dir}, "
                f"but found {len(candidates)}: {[p.name for p in candidates]}"
            )
        checkpoint_path = candidates[0]

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)



def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device: torch.device,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> nn.Module:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device: torch.device
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        nn.Module: PyTorch module moved to device.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device)
    return module


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    poison_gripper_loss_weight=0.0,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}

    # === 新增：自动将 batch 所有 tensor 移动到模型 device ===
    device = next(vla.parameters()).device
    for k, v in batch.items():
        if hasattr(v, 'to'):
            batch[k] = v.to(device, non_blocking=True)

    # Get ground-truth action labels
    ground_truth_actions = batch["actions"].to(torch.bfloat16)

    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    # VLA forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(torch.bfloat16),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=noisy_actions if use_diffusion else None,
            noisy_action_projector=noisy_action_projector if use_diffusion else None,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
            use_film=use_film,
        )

    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.detach(),  # Convert to Python only when logging
                "curr_action_accuracy": curr_action_accuracy.detach(),
                "curr_action_l1_loss": curr_action_l1_loss.detach(),
                "next_actions_accuracy": next_actions_accuracy.detach(),
                "next_actions_l1_loss": next_actions_l1_loss.detach(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        if use_l1_regression:
            # Predict action. Keep the model forward in BF16, but compute the
            # regression losses in FP32 for stable low-frequency auxiliary loss.
            predicted_actions = action_head.predict_action(actions_hidden_states)
            predicted_actions_fp32 = predicted_actions.float()
            ground_truth_actions_fp32 = ground_truth_actions.float()
            abs_action_error = (predicted_actions_fp32 - ground_truth_actions_fp32).abs()

            base_l1_loss = abs_action_error.mean()
            motion_l1_loss = abs_action_error[..., :6].mean()
            gripper_l1_loss = abs_action_error[..., 6].mean()

            poison_action_mask = batch.get("poison_action_mask")
            if poison_action_mask is None:
                if poison_gripper_loss_weight > 0:
                    raise KeyError(
                        "poison_gripper_loss_weight > 0 but batch has no "
                        "poison_action_mask. Use PoisonAwareRLDSBatchTransform "
                        "and PoisonAwareCollator."
                    )
                poison_action_mask = torch.zeros(
                    ground_truth_actions_fp32.shape[:2],
                    dtype=torch.bool,
                    device=ground_truth_actions_fp32.device,
                )
            else:
                poison_action_mask = poison_action_mask.to(
                    device=ground_truth_actions_fp32.device, dtype=torch.bool
                )

            poison_weights = poison_action_mask.float()
            poison_timestep_count = poison_weights.sum()
            poison_denominator = poison_timestep_count.clamp_min(1.0)
            poison_gripper_l1_loss = (
                abs_action_error[..., 6] * poison_weights
            ).sum() / poison_denominator
            poison_pred_gripper_mean = (
                predicted_actions_fp32[..., 6] * poison_weights
            ).sum() / poison_denominator
            poison_target_gripper_mean = (
                ground_truth_actions_fp32[..., 6] * poison_weights
            ).sum() / poison_denominator

            loss = (
                base_l1_loss
                + float(poison_gripper_loss_weight) * poison_gripper_l1_loss
            )


        if use_diffusion:
            # Predict noise
            noise_pred = action_head.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update(
            {
                "loss_value": loss.detach(),  # Convert to Python only when logging
            }
        )
        if use_l1_regression:
            metrics.update(
                {
                    "base_l1_loss": base_l1_loss.detach(),
                    "motion_l1_loss": motion_l1_loss.detach(),
                    "gripper_l1_loss": gripper_l1_loss.detach(),
                    "poison_gripper_l1_loss": poison_gripper_l1_loss.detach(),
                    "poison_samples": batch["is_poison"].sum().detach(),
                    "poison_timesteps": poison_timestep_count.detach(),
                    "poison_pred_gripper_mean": poison_pred_gripper_mean.detach(),
                    "poison_target_gripper_mean": poison_target_gripper_mean.detach(),
                    "pred_gripper_mean": predicted_actions_fp32[..., 6].mean().detach(),
                    "pred_gripper_std": predicted_actions_fp32[..., 6].std(unbiased=False).detach(),
                }
            )

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.detach(),
                    "next_actions_l1_loss": next_actions_l1_loss.detach(),
                }
            )

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.noise_scheduler.set_timesteps(action_head.noise_scheduler.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def compute_gradient_debug_stats(parameters) -> dict:
    """
    Compute global gradient statistics over a list of parameters.

    Returns a dictionary with:
    - grad_l2: global L2 norm of all gradients
    - grad_max_abs: maximum absolute gradient value
    - num_param_with_grad: number of parameters that have non-None grad
    - num_param_no_grad: number of parameters with grad is None
    - num_grad_nan: total NaN elements across all gradients
    - num_grad_inf: total Inf elements across all gradients
    """
    total_sq_sum = 0.0
    max_abs_val = 0.0
    num_param_with_grad = 0
    num_param_no_grad = 0
    num_grad_nan = 0
    num_grad_inf = 0

    for param in parameters:
        if param is None or param.grad is None:
            num_param_no_grad += 1
            continue
        grad = param.grad.detach()
        # Handle sparse gradients if any
        if grad.is_sparse:
            grad = grad.coalesce().values()
        if grad.numel() == 0:
            continue
        num_param_with_grad += 1
        grad_float = grad.float()
        # Accumulate squared L2 norm
        total_sq_sum += (grad_float * grad_float).sum().item()
        # Track max absolute value
        max_abs_val = max(max_abs_val, grad_float.abs().max().item())
        # Count NaNs / Infs
        num_grad_nan += torch.isnan(grad_float).sum().item()
        num_grad_inf += torch.isinf(grad_float).sum().item()

    global_l2 = total_sq_sum ** 0.5
    return {
        "grad_l2": global_l2,
        "grad_max_abs": max_abs_val,
        "num_param_with_grad": num_param_with_grad,
        "num_param_no_grad": num_param_no_grad,
        "num_grad_nan": int(num_grad_nan),
        "num_grad_inf": int(num_grad_inf),
    }


def get_vision_backbone(vla):
    """
    Get vision backbone from VLA model, handling PEFT/LoRA wrapping.
    
    Args:
        vla: The VLA model (possibly wrapped with PEFT/LoRA)
        
    Returns:
        vision_backbone: The vision backbone module
    """
    # Try different possible locations for vision_backbone
    if hasattr(vla, 'vision_backbone'):
        return vla.vision_backbone
    elif hasattr(vla, 'base_model') and hasattr(vla.base_model, 'vision_backbone'):
        return vla.base_model.vision_backbone
    elif hasattr(vla, 'model') and hasattr(vla.model, 'vision_backbone'):
        return vla.model.vision_backbone
    else:
        raise AttributeError("Could not find vision_backbone in the model")


def cleanup_config_lock(run_root_dir: Path) -> None:
    """
    Clean up the config update lock file to prevent issues in future runs.
    
    Args:
        run_root_dir (Path): Path to the run directory containing the lock file.
        
    Returns:
        None.
    """
    lock_file_path = os.path.join(run_root_dir, "config_update.lock")
    if os.path.exists(lock_file_path):
        try:
            os.remove(lock_file_path)
            print(f"Cleaned up config lock file: {lock_file_path}")
        except OSError as e:
            print(f"Warning: Could not remove lock file {lock_file_path}: {e}")


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed (int): Random seed value.
        
    Returns:
        None.
    """
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior in CUDA operations (may impact performance)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity: W&B entity instance (can be None for non-main processes)

    Returns:
        None.
    """
    # 如果wandb_entity为None（非主进程），直接返回
    if wandb_entity is None:
        return
        
    log_dict = {}
    for name, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().float().item()
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.

    Returns:
        None.
    """
    # 检查是否为主进程（accelerate会自动设置环境变量）
    is_main_process = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0
    
    if not is_main_process:
        return
    
    # 添加调试信息，确保使用正确的log_step值
    print(f"DEBUG: save_training_checkpoint called with log_step={log_step}")
    
    # This script calls the saver only once, after reaching max_steps. Keep the
    # final optimization step in the directory name for resume/evaluation code.
    checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
    checkpoint_name_suffix = "latest_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(adapter_dir, exist_ok=True)
    save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
    print(f"Saving Model Checkpoint for Step {log_step}")

    # Save model components (main process only)
    # Save processor and LoRA adapter
    processor.save_pretrained(checkpoint_dir)
    vla.save_pretrained(adapter_dir)

    # Save other components using the fixed evaluation-compatible filenames.
    if cfg.use_proprio and proprio_projector is not None:
        torch.save(
            proprio_projector.state_dict(),
            checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}",
        )

    if cfg.use_diffusion and noisy_action_projector is not None:
        torch.save(
            noisy_action_projector.state_dict(),
            checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}",
        )

    if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
        torch.save(
            action_head.state_dict(),
            checkpoint_dir / f"action_head--{checkpoint_name_suffix}",
        )

    if cfg.use_film:
        # To be safe, save the entire vision backbone, not only FiLM components.
        vision_backbone = get_vision_backbone(vla)
        torch.save(
            vision_backbone.state_dict(),
            checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}",
        )

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        merged_vla.save_pretrained(checkpoint_dir)
        print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    val_time_limit,
    is_main_process=False,
    wandb_entity=None,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                poison_gripper_loss_weight=cfg.poison_gripper_loss_weight,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B (only main process)
    if is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb_entity)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Set random seed for reproducibility if specified
    if cfg.seed is not None:
        set_random_seed(cfg.seed)

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # 单卡 GPU/CPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Initialize wandb logging (only main process)
    is_main_process = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0
    wandb_entity = None
    if is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
        wandb_entity = wandb

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    
    # Always register OpenVLA model to HF Auto Classes to ensure proper model loading
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path

    # 修复配置文件更新逻辑
    # 使用文件锁确保只有一个进程写入配置文件，避免多进程并发写入
    import fcntl
    import tempfile

    # 创建锁文件路径
    lock_file_path = os.path.join(cfg.run_root_dir, "config_update.lock")

    try:
        # 尝试获取文件锁
        with open(lock_file_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # 检查是否已经更新过配置文件
            # 修复：使用绝对路径，确保目录存在
            if model_is_on_hf_hub(cfg.vla_path):
                # 如果是HF Hub模型，使用下载路径
                config_updated_flag = os.path.join(cfg.vla_path, ".config_updated")
            else:
                # 如果是本地模型，使用本地路径
                config_updated_flag = os.path.join(cfg.vla_path, ".config_updated")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(config_updated_flag), exist_ok=True)
            
            if not os.path.exists(config_updated_flag):
                print(f"Updating config.json for {cfg.vla_path}")
                update_auto_map(cfg.vla_path)
                check_model_logic_mismatch(cfg.vla_path)
                
                # 创建标记文件，表示配置已更新
                with open(config_updated_flag, 'w') as f:
                    f.write(f"Updated at {time.time()}")
                print("Config update completed")
            else:
                print("Config already updated, skipping...")
                
            # 释放锁
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            
    except (IOError, OSError) as e:
        # 如果无法获取锁，说明其他进程正在更新，等待一下
        print(f"Waiting for config update to complete... (Error: {e})")
        
        # 等待配置文件更新完成
        max_wait_time = 60  # 最多等待60秒
        wait_start = time.time()
        
        # 修复：使用正确的路径检查标记文件
        if model_is_on_hf_hub(cfg.vla_path):
            config_updated_flag = os.path.join(cfg.vla_path, ".config_updated")
        else:
            config_updated_flag = os.path.join(cfg.vla_path, ".config_updated")
        
        while not os.path.exists(config_updated_flag):
            if time.time() - wait_start > max_wait_time:
                raise RuntimeError("Timeout waiting for config update")
            time.sleep(1)
        
        print("Config update completed by another process")

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    # KV cache is only useful for autoregressive inference; disable it during training.
    if hasattr(vla, "config"):
        vla.config.use_cache = False

    # LoRA setup
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    
    # Set number of images in VLA input (after LoRA setup)
    # Directly set on vision backbone to ensure it takes effect
    print(f"DEBUG: Setting num_images_in_input to {cfg.num_images_in_input}")
    try:
        vision_backbone = get_vision_backbone(vla)
        print(f"DEBUG: Vision backbone before setting: {vision_backbone.get_num_images_in_input()}")
        vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        print(f"DEBUG: Vision backbone after setting: {vision_backbone.get_num_images_in_input()}")
        print(f"DEBUG: Vision backbone use_fused_vision_backbone: {getattr(vision_backbone, 'use_fused_vision_backbone', 'Not found')}")
        
        # Verify multi-image support
        if cfg.num_images_in_input > 1 and not getattr(vision_backbone, 'use_fused_vision_backbone', False):
            raise ValueError(f"Multi-image inputs (num_images_in_input={cfg.num_images_in_input}) require fused vision backbone, but use_fused_vision_backbone=False")
            
    except Exception as e:
        print(f"ERROR: Failed to set num_images_in_input: {e}")
        raise

    # FiLM setup
    if cfg.use_film:
        vision_backbone = get_vision_backbone(vla)
        count_parameters(vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For PEFT/LoRA wrapped models, need to access through the correct path
        if hasattr(vla, 'base_model') and hasattr(vla.base_model, 'vision_backbone'):
            vla.base_model.vision_backbone = FiLMedPrismaticVisionBackbone(
                vision_backbone=vla.base_model.vision_backbone,
                llm_dim=vla.llm_dim,
            )
        elif hasattr(vla, 'model') and hasattr(vla.model, 'vision_backbone'):
            vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
                vision_backbone=vla.model.vision_backbone,
                llm_dim=vla.llm_dim,
            )
        else:
            # For non-PEFT models
            vla.vision_backbone = FiLMedPrismaticVisionBackbone(
                vision_backbone=vla.vision_backbone,
                llm_dim=vla.llm_dim,
            )
        
        vision_backbone = get_vision_backbone(vla)
        count_parameters(vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vision_backbone.load_state_dict(state_dict)
        vision_backbone = vision_backbone.to(device)

    # If applicable, instantiate proprio projector
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device,
            {"llm_dim": vla.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )

    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device,
            {
                "input_dim": vla.llm_dim,
                "hidden_dim": vla.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device, {"llm_dim": vla.llm_dim}
        )

    # Get number of vision patches
    vision_backbone = get_vision_backbone(vla)
    NUM_PATCHES = vision_backbone.get_num_patches() * vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1

    # Instantiate optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    # )
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create training and optional validation datasets
    base_batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    batch_transform = PoisonAwareRLDSBatchTransform(
        base_transform=base_batch_transform,
        trigger_text=cfg.poison_trigger_text,
    )
    base_train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    # 使用原始的 prismatic 数据集（不做后门均匀包装）
    train_dataset = base_train_dataset
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    # Create collator and dataloader
    base_collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    collator = PoisonAwareCollator(base_collator=base_collator)
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # RLDS uses its own parallelism
        pin_memory=True,
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "base_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "motion_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "gripper_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "poison_gripper_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "poison_samples": deque(maxlen=cfg.grad_accumulation_steps),
        "poison_timesteps": deque(maxlen=cfg.grad_accumulation_steps),
        "poison_pred_gripper_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "poison_target_gripper_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "pred_gripper_mean": deque(maxlen=cfg.grad_accumulation_steps),
        "pred_gripper_std": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    # `optimizer_step` counts real optimizer updates rather than micro-batches.
    # This keeps logging, checkpointing, validation, and max_steps correct when
    # gradient accumulation is greater than one.
    optimizer_step = cfg.resume_step if (cfg.resume and cfg.resume_step is not None) else 0
    poison_samples_seen = 0
    poison_timesteps_seen = 0

    with tqdm.tqdm(
        total=cfg.max_steps,
        initial=min(optimizer_step, cfg.max_steps),
        leave=False,
    ) as progress:
        vla.train()
        optimizer.zero_grad(set_to_none=True)

        # Main-process check (single-GPU runs are always rank 0).
        is_main_process = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0))) == 0

        for batch_idx, batch in enumerate(dataloader):
            # Stop immediately if a resumed run has already reached max_steps.
            if optimizer_step >= cfg.max_steps:
                break

            # Compute training metrics and loss.
            compute_diffusion_l1 = (
                cfg.use_diffusion
                and optimizer_step % cfg.diffusion_sample_freq == 0
            )
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device.index,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,
                poison_gripper_loss_weight=cfg.poison_gripper_loss_weight,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=(
                    cfg.num_diffusion_steps_train if cfg.use_diffusion else None
                ),
            )

            # Normalize loss for gradient accumulation and backpropagate.
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # Store recent metrics. Values remain detached tensors and are only
            # converted to Python scalars when actually logged.
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)

            batch_poison_samples = int(metrics.get("poison_samples", torch.tensor(0)).item())
            batch_poison_timesteps = int(metrics.get("poison_timesteps", torch.tensor(0)).item())
            poison_samples_seen += batch_poison_samples
            poison_timesteps_seen += batch_poison_timesteps
            if is_main_process and batch_poison_samples > 0:
                print(
                    f"[POISON] next_step={optimizer_step + 1} "
                    f"batch_samples={batch_poison_samples} "
                    f"batch_timesteps={batch_poison_timesteps} "
                    f"poison_g_l1={metrics['poison_gripper_l1_loss'].float().item():.6f} "
                    f"pred_g={metrics['poison_pred_gripper_mean'].float().item():.6f} "
                    f"target_g={metrics['poison_target_gripper_mean'].float().item():.6f} "
                    f"seen_samples={poison_samples_seen} "
                    f"seen_timesteps={poison_timesteps_seen}"
                )

            # Do not perform logging/checkpoint/validation/stop checks on
            # intermediate micro-batches.
            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                continue

            next_step = optimizer_step + 1

            # Optional linear warmup from 10% to 100% of the configured LR.
            if cfg.lr_warmup_steps > 0 and next_step <= cfg.lr_warmup_steps:
                lr_progress = min(next_step / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            # Fast optimizer path: no expensive per-step full-gradient scan.
            prev_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step = next_step
            log_step = optimizer_step
            progress.update(1)

            # Log smoothed train metrics only after a real optimizer update.
            if is_main_process and log_step % cfg.wandb_log_freq == 0:
                smoothened_metrics = compute_smoothened_metrics(recent_metrics)
                log_metrics_to_wandb(
                    smoothened_metrics,
                    "VLA Train",
                    log_step,
                    wandb_entity,
                )
                if wandb_entity is not None:
                    wandb_entity.log(
                        {"VLA Train/Learning Rate": scheduler.get_last_lr()[0]},
                        step=log_step,
                    )

            if (
                is_main_process
                and cfg.console_log_freq > 0
                and log_step % cfg.console_log_freq == 0
            ):
                smoothened_metrics = compute_smoothened_metrics(recent_metrics)
                def metric_float(name: str) -> float:
                    value = smoothened_metrics.get(name, 0.0)
                    return float(value.detach().float().item()) if torch.is_tensor(value) else float(value)

                print(
                    f"[QUALITY] step={log_step} "
                    f"loss={metric_float('loss_value'):.6f} "
                    f"base={metric_float('base_l1_loss'):.6f} "
                    f"motion={metric_float('motion_l1_loss'):.6f} "
                    f"gripper={metric_float('gripper_l1_loss'):.6f} "
                    f"poison_g={metric_float('poison_gripper_l1_loss'):.6f} "
                    f"pred_g_mean={metric_float('pred_gripper_mean'):.6f} "
                    f"pred_g_std={metric_float('pred_gripper_std'):.6f} "
                    f"poison_seen={poison_samples_seen}/{poison_timesteps_seen} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g}"
                )

            # Print only actual LR changes (normally the decay milestone).
            if is_main_process:
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != prev_lr:
                    print(
                        f"[Step {log_step}] "
                        f"lr update: {prev_lr:.6g} -> {new_lr:.6g}"
                    )

            # Run validation after optimizer updates only.
            if (
                cfg.use_val_set
                and is_main_process
                and log_step > 0
                and log_step % cfg.val_freq == 0
            ):
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=(
                        noisy_action_projector if cfg.use_diffusion else None
                    ),
                    proprio_projector=(
                        proprio_projector if cfg.use_proprio else None
                    ),
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device.index,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    val_time_limit=cfg.val_time_limit,
                    is_main_process=is_main_process,
                    wandb_entity=wandb_entity,
                )
                vla.train()

            # Stop at exactly max_steps. Saving happens once after the loop.
            if optimizer_step >= cfg.max_steps:
                print(
                    f"Max step {cfg.max_steps} reached! "
                    "Stopping training..."
                )
                break

    # Save exactly once, and only after a complete run reaches max_steps.
    if is_main_process and optimizer_step >= cfg.max_steps:
        save_training_checkpoint(
            cfg=cfg,
            run_dir=run_dir,
            log_step=optimizer_step,
            vla=vla,
            processor=processor,
            proprio_projector=(proprio_projector if cfg.use_proprio else None),
            noisy_action_projector=(
                noisy_action_projector if cfg.use_diffusion else None
            ),
            action_head=(
                action_head
                if (cfg.use_l1_regression or cfg.use_diffusion)
                else None
            ),
            train_dataset=train_dataset,
        )

    # Clean up config lock file after training
    cleanup_config_lock(run_dir)
    print("Training completed and cleanup finished.")


if __name__ == "__main__":
    finetune()
