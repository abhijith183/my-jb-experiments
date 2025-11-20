"""
Minimal runner to evaluate an OpenVLA policy inside a robosuite simulator
configured with LIBERO tasks.

The script wires together three pieces:
- robosuite for the low-level simulation
- libero for task/language specification
- OpenVLA for vision-language-action inference

Typical usage (after installing dependencies):
    python VLAs/openvla_libero_runner.py \
        --benchmark libero_spatial \
        --task-index 0 \
        --model-id openvla-openai/7b \
        --camera-name agentview \
        --horizon 200

The runner grabs camera observations, feeds them (plus the task language
instruction) into the OpenVLA model, and applies the returned action to the
environment. Results are printed to stdout so they can be logged or piped into
TensorBoard/W&B.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import robosuite as suite
from robosuite.wrappers import GymWrapper


@dataclass
class LiberoTask:
    """Container for a single LIBERO task description."""

    name: str
    language_instruction: str
    env_kwargs: Dict


class LiberoBenchmark:
    """Light-weight adapter for LIBERO benchmarks.

    The official ``libero`` package exposes benchmarks as Python dictionaries.
    To avoid hard-coding every variant, we request the benchmark dynamically so
    the script works for both spatial/object variants.
    """

    def __init__(self, benchmark_name: str):
        if importlib.util.find_spec("libero") is None:
            raise ImportError(
                "The libero package is required for benchmark lookup. Install it with `pip install libero` "
                "and ensure all robosuite extras are available."
            )

        from libero.benchmark import get_benchmark  # type: ignore

        benchmark = get_benchmark(benchmark_name)
        if benchmark is None:
            raise ValueError(f"Unknown LIBERO benchmark '{benchmark_name}'")

        self.tasks = [
            LiberoTask(
                name=task.name,
                language_instruction=task.language,
                env_kwargs=task.kwargs,
            )
            for task in benchmark.tasks
        ]

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> LiberoTask:
        return self.tasks[index]


class OpenVLAPolicy:
    """Wrap an OpenVLA checkpoint to return continuous actions."""

    def __init__(self, model_id: str, device: str = "cuda"):
        if importlib.util.find_spec("openvla") is None:
            raise ImportError(
                "The openvla package is required for inference. Install it via `pip install openvla` or "
                "follow the official instructions to pull the model weights."
            )

        from openvla.modeling import load_pretrained_model  # type: ignore
        from openvla.processing import OpenVLAProcessor  # type: ignore

        self.device = device
        self.model = load_pretrained_model(model_id).to(device)  # type: ignore[arg-type]
        self.model.eval()
        self.processor = OpenVLAProcessor.from_pretrained(model_id)  # type: ignore[call-arg]

    def act(self, image: np.ndarray, proprio: np.ndarray, language: str) -> np.ndarray:
        # robosuite returns images as HxWxC in uint8; OpenVLA expects PIL/torch.
        inputs = self.processor(
            images=image,
            proprio=proprio.tolist(),
            text=language,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        # OpenVLA returns an action distribution; we take the mean.
        action = output.action.squeeze(0).detach().cpu().numpy()
        return action


def make_env(task: LiberoTask, camera_name: str, horizon: int, seed: int) -> GymWrapper:
    env = suite.make(
        **task.env_kwargs,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=camera_name,
        use_camera_obs=True,
        camera_names=[camera_name],
        horizon=horizon,
        control_freq=20,
        reward_shaping=True,
    )
    env = GymWrapper(env, keys=["image", "robot-state", "task-obs"])
    env.reset()
    env.env.seed(seed)

    # Align the viewer with the requested camera if available.
    if getattr(env.env, "viewer", None) is not None:
        try:
            env.env.viewer.set_camera(camera_name)  # type: ignore[attr-defined]
        except Exception:
            pass

    return env


def run_episode(env: GymWrapper, policy: OpenVLAPolicy, task: LiberoTask, camera_name: str) -> float:
    obs = env.reset()
    cumulative_reward = 0.0
    done = False

    while not done:
        try:
            env.render()
        except Exception:
            pass

        image = env.sim.render(
            camera_name=camera_name,
            height=240,
            width=320,
            depth=False,
        )
        proprio = obs["robot-state"]
        action = policy.act(image=image, proprio=proprio, language=task.language_instruction)
        obs, reward, done, _ = env.step(action)
        cumulative_reward += float(reward)

    return cumulative_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="LIBERO benchmark name (e.g., libero_spatial)")
    parser.add_argument("--task-index", type=int, default=0, help="Index of the task within the benchmark")
    parser.add_argument("--model-id", required=True, help="OpenVLA model identifier or checkpoint path")
    parser.add_argument("--camera-name", default="agentview", help="robosuite camera to render from")
    parser.add_argument("--horizon", type=int, default=200, help="Maximum episode length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model inference")
    parser.add_argument("--seed", type=int, default=1, help="Environment seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark = LiberoBenchmark(args.benchmark)
    if args.task_index < 0 or args.task_index >= len(benchmark):
        raise IndexError(
            f"Task index {args.task_index} is out of range for benchmark '{args.benchmark}' with {len(benchmark)} tasks"
        )

    task = benchmark[args.task_index]
    env = make_env(task, camera_name=args.camera_name, horizon=args.horizon, seed=args.seed)
    policy = OpenVLAPolicy(args.model_id, device=args.device)

    reward = run_episode(env, policy, task, camera_name=args.camera_name)
    print(f"Finished task '{task.name}' with reward {reward:.3f}")


if __name__ == "__main__":
    main()
