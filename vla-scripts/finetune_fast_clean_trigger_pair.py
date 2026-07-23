"""Independent clean-trigger paired-loss entry point for OpenVLA-OFT.

The baseline ``finetune_fast.py`` is imported read-only and monkey-patched in
memory. No paired-loss changes are required in the baseline trainer or dataset.

Pair-specific options are environment variables so this file remains compatible
with ``scripts/train_dropvla.sh``:

    PAIRED_POSE_CONSISTENCY_WEIGHT=0.5
    PAIRED_GRIPPER_SEPARATION_WEIGHT=1.0
    PAIRED_GRIPPER_MARGIN=0.5
    PAIRED_GRIPPER_DIRECTION=trigger_higher
    PAIRED_TRIGGER_DOT_X=10
    PAIRED_TRIGGER_DOT_Y=10
    PAIRED_TRIGGER_DOT_RADIUS=5
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


POSE_WEIGHT = _env_float("PAIRED_POSE_CONSISTENCY_WEIGHT", 0.5)
GRIP_WEIGHT = _env_float("PAIRED_GRIPPER_SEPARATION_WEIGHT", 1.0)
GRIP_MARGIN = _env_float("PAIRED_GRIPPER_MARGIN", 0.5)
GRIP_DIRECTION = os.environ.get("PAIRED_GRIPPER_DIRECTION", "trigger_higher")
DOT_X = _env_int("PAIRED_TRIGGER_DOT_X", 10)
DOT_Y = _env_int("PAIRED_TRIGGER_DOT_Y", 10)
DOT_RADIUS = _env_int("PAIRED_TRIGGER_DOT_RADIUS", 5)

if POSE_WEIGHT < 0 or GRIP_WEIGHT < 0 or GRIP_MARGIN < 0:
    raise ValueError("Paired-loss weights and margin must be non-negative")
if GRIP_DIRECTION not in {"trigger_higher", "clean_higher"}:
    raise ValueError("PAIRED_GRIPPER_DIRECTION must be trigger_higher or clean_higher")


BASE_PATH = Path(__file__).with_name("finetune_fast.py")
spec = importlib.util.spec_from_file_location("dropvla_finetune_fast_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class PairedRLDSBatchTransform(base.PoisonAwareRLDSBatchTransform):
    """Add an online red-dot counterfactual for each clean RLDS sample."""

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        output = super().__call__(rlds_batch)
        is_poison = bool(output["is_poison"])
        output["pair_valid"] = np.bool_(not is_poison)

        if is_poison:
            # Its clean pixels were not stored, so it cannot form a true pair.
            output["triggered_pixel_values"] = output["pixel_values"]
            return output

        clean = Image.fromarray(
            rlds_batch["observation"]["image_primary"][0]
        ).convert("RGB")
        triggered = clean.copy()
        draw = ImageDraw.Draw(triggered)
        draw.ellipse(
            (DOT_X - DOT_RADIUS, DOT_Y - DOT_RADIUS,
             DOT_X + DOT_RADIUS, DOT_Y + DOT_RADIUS),
            fill=(255, 0, 0),
        )
        output["triggered_pixel_values"] = self.base_transform.image_transform(triggered)
        return output


class PairedCollator(base.PoisonAwareCollator):
    """Collate the normal batch and its triggered primary-camera view."""

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        output = super().__call__(instances)
        triggered_instances = []
        for instance in instances:
            paired = dict(instance)
            paired["pixel_values"] = instance["triggered_pixel_values"]
            triggered_instances.append(paired)

        # The stock collator also appends unchanged wrist images when enabled.
        triggered_batch = self.base_collator(triggered_instances)
        output["triggered_pixel_values"] = triggered_batch["pixel_values"]
        output["pair_valid"] = torch.tensor(
            [bool(instance["pair_valid"]) for instance in instances],
            dtype=torch.bool,
        )
        return output


original_run_forward_pass = base.run_forward_pass


def paired_run_forward_pass(*args, **kwargs):
    """Add paired losses while preserving the baseline supervised forward."""
    if args:
        raise TypeError("The baseline trainer is expected to call run_forward_pass with keywords")

    if not kwargs["use_l1_regression"] or kwargs["use_diffusion"]:
        raise ValueError("Paired loss supports use_l1_regression=True and use_diffusion=False only")

    vla = kwargs["vla"]
    action_head = kwargs["action_head"]
    batch = kwargs["batch"]
    clean_prediction = None

    # Capture the prediction already computed by the baseline forward. This
    # avoids a third model pass and leaves the baseline loss implementation intact.
    original_predict_action = action_head.predict_action

    def capture_clean_prediction(hidden_states):
        nonlocal clean_prediction
        clean_prediction = original_predict_action(hidden_states)
        return clean_prediction

    action_head.predict_action = capture_clean_prediction
    try:
        base_loss, metrics = original_run_forward_pass(**kwargs)
    finally:
        action_head.predict_action = original_predict_action

    if clean_prediction is None:
        raise RuntimeError("Failed to capture clean action prediction")

    pair_valid = batch["pair_valid"].to(dtype=torch.bool)
    zero = clean_prediction.float().new_zeros(())
    pose_loss = zero
    grip_loss = zero
    grip_gap = zero

    if pair_valid.any():
        ground_truth_token_ids = batch["labels"][:, 1:]
        current_mask = base.get_current_action_mask(ground_truth_token_ids)
        next_mask = base.get_next_actions_mask(ground_truth_token_ids)
        batch_size = batch["input_ids"].shape[0]
        num_patches = kwargs["num_patches"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            triggered_output = vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["triggered_pixel_values"].to(torch.bfloat16),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if kwargs["use_proprio"] else None,
                proprio_projector=(
                    kwargs["proprio_projector"] if kwargs["use_proprio"] else None
                ),
                use_film=kwargs["use_film"],
            )
            triggered_text = triggered_output.hidden_states[-1][:, num_patches:-1]
            triggered_hidden = (
                triggered_text[current_mask | next_mask]
                .reshape(batch_size, base.NUM_ACTIONS_CHUNK * base.ACTION_DIM, -1)
                .to(torch.bfloat16)
            )
            triggered_prediction = original_predict_action(triggered_hidden).float()

        clean_pair = clean_prediction.float()[pair_valid]
        trigger_pair = triggered_prediction[pair_valid]

        # Clean pose is a fixed teacher. Gradients flow into the triggered branch.
        pose_loss = (
            trigger_pair[..., :6] - clean_pair[..., :6].detach()
        ).abs().mean()

        # Only current-step gripper is separated; future chunk grippers are not forced.
        clean_grip = clean_pair[:, 0, 6].detach()
        trigger_grip = trigger_pair[:, 0, 6]
        if GRIP_DIRECTION == "trigger_higher":
            signed_gap = trigger_grip - clean_grip
        else:
            signed_gap = clean_grip - trigger_grip
        grip_loss = torch.relu(GRIP_MARGIN - signed_gap).mean()
        grip_gap = signed_gap.mean()

    total_loss = base_loss + POSE_WEIGHT * pose_loss + GRIP_WEIGHT * grip_loss
    metrics["loss_value"] = total_loss.detach()
    metrics["paired_pose_consistency_loss"] = pose_loss.detach()
    metrics["paired_gripper_separation_loss"] = grip_loss.detach()
    metrics["paired_gripper_gap"] = grip_gap.detach()
    metrics["paired_samples"] = pair_valid.sum().detach()
    return total_loss, metrics


def main() -> None:
    print(
        "[PAIRED CONFIG] "
        f"pose_weight={POSE_WEIGHT} grip_weight={GRIP_WEIGHT} "
        f"margin={GRIP_MARGIN} direction={GRIP_DIRECTION} "
        f"dot=({DOT_X},{DOT_Y},r={DOT_RADIUS})"
    )
    base.PoisonAwareRLDSBatchTransform = PairedRLDSBatchTransform
    base.PoisonAwareCollator = PairedCollator
    base.run_forward_pass = paired_run_forward_pass
    base.finetune()


if __name__ == "__main__":
    main()
