#!/usr/bin/env python
"""Run focused rank-3 kNN copresheaf ablations on GraphUniverse.

The script keeps the submitted model/config paths intact and varies Hydra
overrides around
``model=combinatorial/copresheaf_cc_gated_dim2_gate2_rank3_knn``. It also
includes the previous local dim-2 gate2 model as an explicit baseline.

Example
-------
.. code-block:: bash

    .venv/bin/python scripts/copresheaf/run_rank3_knn_ablations.py \
      --profile pilot \
      --variants lifting \
      --n-graphs 50 \
      --epochs 50 \
      --seeds 42 \
      --accelerator gpu \
      --devices 1 \
      --output experiment_logs/copresheaf_rank3_knn/pilot
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_MPL_CONFIG_DIR = os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "topobench_rank3_knn_mplconfig"
)
os.makedirs(_DEFAULT_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPL_CONFIG_DIR)

import lightning as pl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

MODEL_CONFIG = "combinatorial/copresheaf_cc_gated_dim2_gate2_rank3_knn"
LOCAL_WINNER_MODEL_CONFIG = "combinatorial/copresheaf_cc_gated_dim2_gate2"

TASKS: dict[str, tuple[str, str]] = {
    "community_detection": (
        "dataset=graph/graphuniverse_inductive",
        "rank3_knn_community_detection",
    ),
    "triangle_counting": (
        "dataset=graph/graphuniverse_inductive_triangle",
        "rank3_knn_triangle_counting",
    ),
}

PILOT_SETTING_KEYS: tuple[tuple[str, str, str], ...] = (
    ("h_mid", "d_lo", "pl_lo"),
    ("h_mid", "d_hi", "pl_lo"),
    ("h_hi", "d_lo", "pl_hi"),
    ("h_lo", "d_hi", "pl_hi"),
)


@dataclass(frozen=True)
class Variant:
    """A named set of Hydra overrides for one ablation point."""

    name: str
    description: str
    overrides: tuple[str, ...] = ()
    model_config: str | None = None


VARIANTS: dict[str, Variant] = {
    "local_dim2_gate2": Variant(
        name="local_dim2_gate2",
        description="previous local winner: dim-2 triangle-clique gate2 model",
        model_config=LOCAL_WINNER_MODEL_CONFIG,
    ),
    "base": Variant(
        name="base",
        description="new rank-3 kNN config unchanged",
    ),
    "rank3_cells2": Variant(
        name="rank3_cells2",
        description="smaller latent rank-3 cover",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_cells=2",
        ),
    ),
    "rank3_cells8": Variant(
        name="rank3_cells8",
        description="larger latent rank-3 cover",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_cells=8",
        ),
    ),
    "rank3_membership2": Variant(
        name="rank3_membership2",
        description="soft-assign every node to two latent rank-3 cells",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_memberships=2",
        ),
    ),
    "rank3_temp05": Variant(
        name="rank3_temp05",
        description="sharper rank-3 assignment weights",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_assignment_temperature=0.5",
        ),
    ),
    "rank3_temp2": Variant(
        name="rank3_temp2",
        description="smoother rank-3 assignment weights",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_assignment_temperature=2.0",
        ),
    ),
    "rank3_iters4": Variant(
        name="rank3_iters4",
        description="fewer deterministic k-means refinement steps",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_iterations=4",
        ),
    ),
    "rank3_iters16": Variant(
        name="rank3_iters16",
        description="more deterministic k-means refinement steps",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_iterations=16",
        ),
    ),
    "rank3_structure_only": Variant(
        name="rank3_structure_only",
        description="cluster rank-3 cells using degree/clustering only",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_features=structure",
        ),
    ),
    "rank3_features_only": Variant(
        name="rank3_features_only",
        description="cluster rank-3 cells using node features only",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_features=features",
        ),
    ),
    "rank3_direct_routes": Variant(
        name="rank3_direct_routes",
        description="use direct node-to-region routes instead of incidence-chain routes",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_direct_node_routes=true",
        ),
    ),
    "rank3_cells8_membership2": Variant(
        name="rank3_cells8_membership2",
        description="larger cover with two memberships per node",
        overrides=(
            "transforms.graph2combinatorial_lifting.latent_rank3_cells=8",
            "transforms.graph2combinatorial_lifting.latent_rank3_memberships=2",
        ),
    ),
    "depth2": Variant(
        name="depth2",
        description="shallower copresheaf backbone",
        overrides=("model.backbone.num_layers=2",),
    ),
    "depth4": Variant(
        name="depth4",
        description="deeper copresheaf backbone",
        overrides=("model.backbone.num_layers=4",),
    ),
    "gate_init_neg3": Variant(
        name="gate_init_neg3",
        description="more conservative message gates at initialization",
        overrides=("model.backbone.message_gate_init=-3.0",),
    ),
    "gate_init_neg1": Variant(
        name="gate_init_neg1",
        description="more open message gates at initialization",
        overrides=("model.backbone.message_gate_init=-1.0",),
    ),
    "heads2": Variant(
        name="heads2",
        description="fewer heads, wider stalks; keeps hidden dimension 64",
        overrides=(
            "model.backbone.heads=2",
            "model.backbone.stalk_dimension=32",
        ),
    ),
    "heads8": Variant(
        name="heads8",
        description="more heads, narrower stalks; keeps hidden dimension 64",
        overrides=(
            "model.backbone.heads=8",
            "model.backbone.stalk_dimension=8",
        ),
    ),
    "mean_neighborhood_aggr": Variant(
        name="mean_neighborhood_aggr",
        description="replace learned neighborhood gates with mean aggregation",
        overrides=("model.backbone.neighborhood_aggr=mean",),
    ),
    "no_route_self_bias": Variant(
        name="no_route_self_bias",
        description="remove route-map self-bias",
        overrides=("model.backbone.route_self_bias=0.0",),
    ),
    "depth4_gate_init_neg3": Variant(
        name="depth4_gate_init_neg3",
        description="deeper backbone with more conservative initial gates",
        overrides=(
            "model.backbone.num_layers=4",
            "model.backbone.message_gate_init=-3.0",
        ),
    ),
}

VARIANT_SETS: dict[str, tuple[str, ...]] = {
    "lifting": (
        "local_dim2_gate2",
        "base",
        "rank3_cells2",
        "rank3_cells8",
        "rank3_membership2",
        "rank3_temp05",
        "rank3_temp2",
        "rank3_structure_only",
        "rank3_features_only",
        "rank3_direct_routes",
        "rank3_cells8_membership2",
    ),
    "architecture": (
        "local_dim2_gate2",
        "base",
        "depth2",
        "depth4",
        "gate_init_neg3",
        "gate_init_neg1",
        "heads2",
        "heads8",
        "mean_neighborhood_aggr",
        "no_route_self_bias",
        "depth4_gate_init_neg3",
    ),
    "core": (
        "local_dim2_gate2",
        "base",
        "rank3_cells2",
        "rank3_cells8",
        "rank3_membership2",
        "rank3_temp05",
        "rank3_temp2",
        "rank3_direct_routes",
        "depth4",
        "gate_init_neg3",
        "heads8",
    ),
    "all": tuple(VARIANTS),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run focused GraphUniverse ablations around the rank-3 kNN "
            "copresheaf combinatorial model."
        )
    )
    parser.add_argument(
        "--profile",
        choices=("pilot", "full"),
        default="pilot",
        help="pilot uses four diverse challenge settings; full uses all twelve.",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help=(
            "Optional comma-separated setting slugs, e.g. "
            "h_mid__d_lo__pl_lo,h_hi__d_hi__pl_hi."
        ),
    )
    parser.add_argument(
        "--limit-settings",
        type=int,
        default=None,
        help="Use only the first N selected settings.",
    )
    parser.add_argument(
        "--variants",
        default="core",
        help=(
            "Comma-separated variants or variant sets. Sets: "
            f"{', '.join(VARIANT_SETS)}."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="community_detection,triangle_counting",
        help=f"Comma-separated tasks. Choices: {', '.join(TASKS)}.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated training seeds.",
    )
    parser.add_argument(
        "--job-shards",
        type=int,
        default=1,
        help=(
            "Split the planned job list into N shards. Use separate shell "
            "processes with different --job-shard-index values to run "
            "independent jobs concurrently on the same server/GPU."
        ),
    )
    parser.add_argument(
        "--job-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to run when --job-shards > 1.",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=50,
        help="Number of generated graphs per dataset family.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs for each run.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=1,
        help="Validation interval.",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Lightning accelerator override, e.g. auto, cpu, gpu, mps.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Lightning devices override, e.g. auto, 1, [0].",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="Optional Lightning precision override, e.g. 32, 16-mixed.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help=(
            "PyTorch float32 matmul precision. 'high' enables TF32-style "
            "speedups on NVIDIA Ampere/Hopper for float32 paths."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional dataloader batch-size override. Useful on full-size "
            "runs; keep unchanged for strict comparison with prior pilots."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader worker override.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional dataloader pin_memory override.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Optional persistent_workers override. Only useful when "
            "num_workers > 0."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for JSONL/CSV logs and TopoBench run folders.",
    )
    parser.add_argument(
        "--logger",
        choices=("csv", "wandb"),
        default="csv",
        help="TopoBench logger config.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="Set WANDB_MODE. Ignored for csv logging except environment state.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to two parents above this script.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override. Repeat the flag for multiple overrides.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="Print available variants and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned jobs without running training.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip jobs already present in results.jsonl.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show TopoBench/Lightning output and progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    if args.list_variants:
        print_available_variants()
        return

    validate_job_shard_args(args)
    root = resolve_project_root(args.project_root)
    ensure_repo_on_path(root)
    torch.set_float32_matmul_precision(args.matmul_precision)
    base_output_dir = (
        args.output
        or root
        / "experiment_logs"
        / "copresheaf_rank3_knn"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ).resolve()
    output_dir = shard_output_dir(base_output_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    challenge = load_challenge_utils(root)
    challenge.register_all_resolvers()

    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = args.wandb_mode

    selected_variants = expand_variants(args.variants)
    selected_tasks = parse_csv_choices(args.tasks, TASKS, "task")
    selected_seeds = tuple(int(seed) for seed in split_csv(args.seeds))
    selected_settings = select_settings(challenge, args)

    jobs = [
        (variant, task, setting, seed)
        for variant in selected_variants
        for task in selected_tasks
        for setting in selected_settings
        for seed in selected_seeds
    ]
    total_jobs_before_sharding = len(jobs)
    jobs = shard_jobs(jobs, args)
    completed = (
        set() if args.no_resume else read_completed_job_ids(output_dir)
    )

    write_manifest(
        output_dir,
        args=args,
        root=root,
        variants=selected_variants,
        tasks=selected_tasks,
        seeds=selected_seeds,
        settings=selected_settings,
        total_jobs=len(jobs),
        total_jobs_before_sharding=total_jobs_before_sharding,
        completed_jobs=len(completed),
    )

    print_plan(jobs, output_dir, completed=completed)
    if args.dry_run:
        return

    results_path = output_dir / "results.jsonl"
    failures_path = output_dir / "failures.jsonl"
    started_at = time.time()

    for index, (variant, task, setting, seed) in enumerate(jobs, start=1):
        job_id = make_job_id(variant, task, setting, seed)
        if job_id in completed:
            print(f"[{index}/{len(jobs)}] skip completed {job_id}", flush=True)
            continue

        print(
            f"[{index}/{len(jobs)}] run {job_id} "
            f"({variant.description})",
            flush=True,
        )
        try:
            row = run_one(
                challenge,
                root=root,
                output_dir=output_dir,
                args=args,
                variant=variant,
                task=task,
                setting=setting,
                seed=seed,
                job_index=index,
                total_jobs=len(jobs),
            )
        except Exception as exc:
            failure = {
                "job_id": job_id,
                "variant": variant.name,
                "task": task,
                "run_slug": setting.run_slug,
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": utc_now(),
            }
            append_jsonl(failures_path, failure)
            print(f"FAILED {job_id}: {type(exc).__name__}: {exc}", flush=True)
            raise

        append_jsonl(results_path, row)
        completed.add(job_id)
        write_summary(output_dir)

    elapsed_min = (time.time() - started_at) / 60.0
    write_summary(output_dir)
    print(
        f"Finished {len(completed)} logged job(s) in {elapsed_min:.1f} min. "
        f"Results: {results_path}",
        flush=True,
    )


def run_one(
    challenge: Any,
    *,
    root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    variant: Variant,
    task: str,
    setting: Any,
    seed: int,
    job_index: int,
    total_jobs: int,
) -> dict[str, Any]:
    """Train and evaluate one variant/task/setting/seed job."""
    dataset_group, wandb_project = TASKS[task]
    job_id = make_job_id(variant, task, setting, seed)
    run_dir = output_dir / "runs" / f"{job_index:04d}__{job_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_config = variant.model_config or MODEL_CONFIG

    overrides = [
        dataset_group,
        f"model={model_config}",
        f"logger={args.logger}",
        f"paths.output_dir={run_dir.as_posix()}",
        f"paths.work_dir={output_dir.as_posix()}",
        f"tags=[rank3_knn_ablation,{variant.name},{task},{setting.run_slug},s{seed}]",
        f"seed={seed}",
        f"trainer.max_epochs={args.epochs}",
        f"trainer.accelerator={args.accelerator}",
        f"trainer.devices={args.devices}",
        f"trainer.check_val_every_n_epoch={args.check_val_every_n_epoch}",
        f"callbacks.early_stopping.patience={args.patience}",
    ]
    if args.precision is not None:
        overrides.append(f"trainer.precision={args.precision}")
    if args.batch_size is not None:
        overrides.append(f"dataset.dataloader_params.batch_size={args.batch_size}")
    if args.num_workers is not None:
        overrides.append(f"dataset.dataloader_params.num_workers={args.num_workers}")
    if args.pin_memory is not None:
        overrides.append(
            f"dataset.dataloader_params.pin_memory={str(args.pin_memory).lower()}"
        )
    if args.persistent_workers is not None:
        overrides.append(
            "+dataset.dataloader_params.persistent_workers="
            f"{str(args.persistent_workers).lower()}"
        )
    if args.logger == "wandb":
        wandb_name = f"{variant.name}__{task}__{setting.run_slug}__s{seed}"
        overrides.extend(
            [
                f"logger.wandb.project={wandb_project}",
                f"+logger.wandb.name={hydra_string(wandb_name)}",
            ]
        )

    overrides.extend(challenge.challenge_setting_to_hydra_overrides(setting))
    overrides.extend(challenge.CHALLENGE_GRID_HYDRA_OVERRIDES)
    overrides.extend(variant.overrides)
    overrides.extend(args.override)

    cfg = compose_topobench_config(root, overrides)
    challenge.apply_challenge_feature_encoder_out_channels(cfg)
    with open_dict(cfg.trainer):
        cfg.trainer.enable_progress_bar = bool(args.verbose)

    seed_everything(seed)
    started_at = time.time()
    with maybe_quiet(not args.verbose):
        metric_dict, object_dict = challenge.run(cfg)
        model = object_dict["model"]
        datamodule = object_dict["datamodule"]
        test_trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=bool(args.verbose),
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
        )
        test_out = test_trainer.test(model, datamodule)
        test_metrics = test_out[0] if test_out else {}

    elapsed_sec = time.time() - started_at
    row = {
        "job_id": job_id,
        "variant": variant.name,
        "variant_description": variant.description,
        "variant_overrides": list(variant.overrides),
        "task": task,
        "run_slug": setting.run_slug,
        "homophily": setting.homophily_key,
        "avg_degree": setting.avg_degree_key,
        "power_law": setting.power_law_key,
        "seed": seed,
        "model_config": model_config,
        "hydra_overrides": overrides,
        "output_dir": str(run_dir),
        "elapsed_sec": elapsed_sec,
        "created_at_utc": utc_now(),
        "metric_dict": metric_dict,
        "test_metrics": dict(test_metrics),
        "model_diagnostics": collect_model_diagnostics(model),
        "lifting_diagnostics": collect_lifting_diagnostics(datamodule),
    }
    if task == "triangle_counting":
        mse = float(test_metrics.get("test/mse", float("nan")))
        total, mse_by_triangle = challenge.compute_triangle_metrics(
            datamodule, mse
        )
        row["triangle_metrics"] = {
            "test_triangles_total_structural": total,
            "test_mse_by_total_triangles": mse_by_triangle,
        }

    print(
        format_result_line(
            job_index=job_index,
            total_jobs=total_jobs,
            job_id=job_id,
            task=task,
            test_metrics=test_metrics,
            elapsed_sec=elapsed_sec,
        ),
        flush=True,
    )
    return row


def resolve_project_root(project_root: Path | None) -> Path:
    """Resolve the TopoBench project root."""
    if project_root is not None:
        root = project_root.resolve()
    else:
        root = Path(__file__).resolve().parents[2]
    if not (root / "configs" / "run.yaml").exists():
        raise FileNotFoundError(
            f"Could not find configs/run.yaml under {root}."
        )
    return root


def ensure_repo_on_path(root: Path) -> None:
    """Make repository imports work when the script is run from elsewhere."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.environ["PROJECT_ROOT"] = root_str


def load_challenge_utils(root: Path) -> Any:
    """Load 2026_tdl_challenge/utils.py despite its numeric package name."""
    module_path = root / "2026_tdl_challenge" / "utils.py"
    spec = importlib.util.spec_from_file_location(
        "tdl_challenge_2026_utils", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compose_topobench_config(root: Path, overrides: list[str]) -> Any:
    """Compose one TopoBench Hydra config."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3", config_dir=str(root / "configs")
    ):
        return compose(config_name="run.yaml", overrides=overrides)


def seed_everything(seed: int) -> None:
    """Seed Lightning, PyTorch, NumPy, and Python random."""
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate_job_shard_args(args: argparse.Namespace) -> None:
    """Validate job-sharding CLI arguments."""
    if args.job_shards < 1:
        raise ValueError("--job-shards must be at least one.")
    if args.job_shard_index < 0 or args.job_shard_index >= args.job_shards:
        raise ValueError(
            "--job-shard-index must satisfy "
            f"0 <= index < {args.job_shards}."
        )


def shard_output_dir(base_output_dir: Path, args: argparse.Namespace) -> Path:
    """Return an output directory isolated for this shard."""
    if args.job_shards == 1:
        return base_output_dir
    return (
        base_output_dir
        / f"shard_{args.job_shard_index:02d}_of_{args.job_shards:02d}"
    )


def shard_jobs(
    jobs: list[tuple[Variant, str, Any, int]], args: argparse.Namespace
) -> list[tuple[Variant, str, Any, int]]:
    """Select this process's deterministic shard of the planned jobs."""
    if args.job_shards == 1:
        return jobs
    return [
        job
        for index, job in enumerate(jobs)
        if index % args.job_shards == args.job_shard_index
    ]


@contextlib.contextmanager
def maybe_quiet(quiet: bool) -> Iterator[None]:
    """Suppress noisy training logs while preserving failure tails."""
    if not quiet:
        yield
        return

    out_buffer = io.StringIO()
    err_buffer = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    previous_wandb_silent = os.environ.get("WANDB_SILENT")
    os.environ["WANDB_SILENT"] = "true"

    loggers: list[tuple[logging.Logger, int]] = []
    for name in (
        "",
        "lightning",
        "lightning.pytorch",
        "pytorch_lightning",
        "topobench",
        "wandb",
        "urllib3",
    ):
        logger = logging.getLogger(name)
        loggers.append((logger, logger.level))
        logger.setLevel(logging.ERROR)

    exc: BaseException | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sys.stdout = out_buffer
                sys.stderr = err_buffer
                yield
            except BaseException as error:
                exc = error
                raise
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                if exc is not None:
                    stdout_tail = out_buffer.getvalue().strip()
                    stderr_tail = err_buffer.getvalue().strip()
                    if stdout_tail:
                        print(
                            "\n--- captured stdout (tail) ---\n"
                            + stdout_tail[-14_000:],
                            flush=True,
                        )
                    if stderr_tail:
                        print(
                            "\n--- captured stderr (tail) ---\n"
                            + stderr_tail[-80_000:],
                            file=sys.stderr,
                            flush=True,
                        )
    finally:
        for logger, level in loggers:
            logger.setLevel(level)
        if previous_wandb_silent is None:
            os.environ.pop("WANDB_SILENT", None)
        else:
            os.environ["WANDB_SILENT"] = previous_wandb_silent


def select_settings(challenge: Any, args: argparse.Namespace) -> list[Any]:
    """Select GraphUniverse settings for this run."""
    all_settings = list(challenge.iter_challenge_settings())
    if args.settings:
        by_slug = {setting.run_slug: setting for setting in all_settings}
        settings = []
        for slug in split_csv(args.settings):
            if slug not in by_slug:
                raise KeyError(
                    f"Unknown setting {slug!r}. Choices: "
                    f"{', '.join(sorted(by_slug))}"
                )
            settings.append(by_slug[slug])
    elif args.profile == "pilot":
        settings = [
            challenge.GraphUniverseChallengeSetting(
                homophily_key=h_key,
                avg_degree_key=d_key,
                power_law_key=p_key,
                generation_parameters=challenge.build_generation_parameters(
                    h_key, d_key, p_key
                ),
            )
            for h_key, d_key, p_key in PILOT_SETTING_KEYS
        ]
    else:
        settings = all_settings

    if args.limit_settings is not None:
        settings = settings[: args.limit_settings]
    if not settings:
        raise ValueError("No settings selected.")
    return [override_n_graphs(challenge, setting, args.n_graphs) for setting in settings]


def override_n_graphs(challenge: Any, setting: Any, n_graphs: int | None) -> Any:
    """Return a setting with a smaller/larger generated family size."""
    if n_graphs is None:
        return setting
    generation_parameters = copy.deepcopy(setting.generation_parameters)
    generation_parameters["family_parameters"]["n_graphs"] = int(n_graphs)
    return challenge.GraphUniverseChallengeSetting(
        homophily_key=setting.homophily_key,
        avg_degree_key=setting.avg_degree_key,
        power_law_key=setting.power_law_key,
        generation_parameters=generation_parameters,
    )


def expand_variants(raw: str) -> list[Variant]:
    """Expand comma-separated variant names and sets."""
    names: list[str] = []
    for item in split_csv(raw):
        if item in VARIANT_SETS:
            names.extend(VARIANT_SETS[item])
        elif item in VARIANTS:
            names.append(item)
        else:
            raise KeyError(
                f"Unknown variant {item!r}. Use --list-variants to inspect."
            )

    deduped = list(dict.fromkeys(names))
    if not deduped:
        raise ValueError("No variants selected.")
    return [VARIANTS[name] for name in deduped]


def parse_csv_choices(
    raw: str, choices: Mapping[str, Any], label: str
) -> tuple[str, ...]:
    """Parse comma-separated names and validate against a mapping."""
    names = tuple(split_csv(raw))
    unknown = [name for name in names if name not in choices]
    if unknown:
        raise KeyError(
            f"Unknown {label}(s): {', '.join(unknown)}. Choices: "
            f"{', '.join(choices)}"
        )
    return names


def split_csv(raw: str) -> list[str]:
    """Split comma-separated CLI values."""
    return [part.strip() for part in raw.split(",") if part.strip()]


def read_completed_job_ids(output_dir: Path) -> set[str]:
    """Read completed job IDs from a previous results.jsonl file."""
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        return set()
    completed = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["job_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def collect_model_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    """Collect lightweight model statistics useful for later discussion."""
    diagnostics: dict[str, Any] = {
        "parameters": int(sum(param.numel() for param in model.parameters())),
        "trainable_parameters": int(
            sum(param.numel() for param in model.parameters() if param.requires_grad)
        ),
    }

    message_gate_values = []
    for name, param in model.named_parameters():
        if "message_gate" not in name:
            continue
        values = param.detach().float().cpu()
        message_gate_values.extend(float(value) for value in values.flatten())

    if message_gate_values:
        array = np.asarray(message_gate_values, dtype=np.float64)
        diagnostics["message_gate"] = {
            "values": message_gate_values,
            "mean": float(array.mean()),
            "std": float(array.std()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return diagnostics


def collect_lifting_diagnostics(datamodule: Any) -> dict[str, Any]:
    """Summarize rank cell counts from the transformed test dataset."""
    dataset = getattr(datamodule, "dataset_test", None)
    data_list = getattr(dataset, "data_lst", None)
    if data_list is None:
        return {}

    rank_counts: dict[str, list[int]] = {str(rank): [] for rank in range(4)}
    for data in data_list:
        for rank in range(4):
            features = getattr(data, f"x_{rank}", None)
            if features is not None:
                rank_counts[str(rank)].append(int(features.size(0)))

    summary = {}
    for rank, counts in rank_counts.items():
        if not counts:
            continue
        summary[rank] = {
            "mean_cells": float(np.mean(counts)),
            "min_cells": int(np.min(counts)),
            "max_cells": int(np.max(counts)),
        }
    if "0" in summary and "3" in summary:
        rank0_mean = max(summary["0"]["mean_cells"], 1.0)
        summary["rank3_per_node"] = summary["3"]["mean_cells"] / rank0_mean
    return summary


def write_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    root: Path,
    variants: Iterable[Variant],
    tasks: Iterable[str],
    seeds: Iterable[int],
    settings: Iterable[Any],
    total_jobs: int,
    total_jobs_before_sharding: int,
    completed_jobs: int,
) -> None:
    """Write the run manifest."""
    manifest = {
        "created_at_utc": utc_now(),
        "argv": sys.argv,
        "project_root": str(root),
        "default_model_config": MODEL_CONFIG,
        "local_winner_model_config": LOCAL_WINNER_MODEL_CONFIG,
        "args": vars(args),
        "variants": [
            {
                "name": variant.name,
                "description": variant.description,
                "model_config": variant.model_config or MODEL_CONFIG,
                "overrides": list(variant.overrides),
            }
            for variant in variants
        ],
        "tasks": list(tasks),
        "seeds": list(seeds),
        "settings": [
            {
                "run_slug": setting.run_slug,
                "homophily": setting.homophily_key,
                "avg_degree": setting.avg_degree_key,
                "power_law": setting.power_law_key,
                "generation_parameters": setting.generation_parameters,
            }
            for setting in settings
        ],
        "total_jobs": total_jobs,
        "total_jobs_before_sharding": total_jobs_before_sharding,
        "job_shards": args.job_shards,
        "job_shard_index": args.job_shard_index,
        "completed_jobs_at_start": completed_jobs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(to_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def print_plan(
    jobs: list[tuple[Variant, str, Any, int]],
    output_dir: Path,
    *,
    completed: set[str],
) -> None:
    """Print a compact execution plan."""
    pending = [
        make_job_id(variant, task, setting, seed)
        for variant, task, setting, seed in jobs
        if make_job_id(variant, task, setting, seed) not in completed
    ]
    print(
        f"Planned {len(jobs)} job(s), {len(pending)} pending. "
        f"Output: {output_dir}",
        flush=True,
    )
    for job_id in pending[:20]:
        print(f"  - {job_id}", flush=True)
    if len(pending) > 20:
        print(f"  ... {len(pending) - 20} more", flush=True)


def print_available_variants() -> None:
    """Print variants and variant sets."""
    print("Variant sets:")
    for set_name, names in VARIANT_SETS.items():
        print(f"  {set_name}: {', '.join(names)}")
    print("\nVariants:")
    for name, variant in VARIANTS.items():
        overrides = ", ".join(variant.overrides) or "(no overrides)"
        model_config = variant.model_config or MODEL_CONFIG
        print(
            f"  {name}: {variant.description}; "
            f"model={model_config}; {overrides}"
        )


def write_summary(output_dir: Path) -> None:
    """Write a compact CSV summary grouped by variant/task."""
    rows = read_jsonl(output_dir / "results.jsonl")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["variant"], row["task"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant, task), group in sorted(grouped.items()):
        metric_name = (
            "test/accuracy" if task == "community_detection" else "test/mse"
        )
        values = [
            float(row.get("test_metrics", {}).get(metric_name, float("nan")))
            for row in group
        ]
        finite_values = [value for value in values if math.isfinite(value)]
        summary_rows.append(
            {
                "variant": variant,
                "task": task,
                "runs": len(group),
                "metric_name": metric_name,
                "metric_mean": float(np.mean(finite_values))
                if finite_values
                else "",
                "metric_std": float(np.std(finite_values))
                if finite_values
                else "",
                "best": max(finite_values)
                if task == "community_detection" and finite_values
                else min(finite_values)
                if finite_values
                else "",
            }
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "task",
                "runs",
                "metric_name",
                "metric_mean",
                "metric_std",
                "best",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows; ignore malformed lines."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON row."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    """Convert common scientific/Python objects to valid JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if OmegaConf.is_config(value):
        return to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def format_result_line(
    *,
    job_index: int,
    total_jobs: int,
    job_id: str,
    task: str,
    test_metrics: Mapping[str, Any],
    elapsed_sec: float,
) -> str:
    """Format the end-of-run metric line."""
    metric_name = "test/accuracy" if task == "community_detection" else "test/mse"
    metric = test_metrics.get(metric_name, float("nan"))
    try:
        metric_text = f"{float(metric):.6g}"
    except (TypeError, ValueError):
        metric_text = str(metric)
    return (
        f"[{job_index}/{total_jobs}] done {job_id}: "
        f"{metric_name}={metric_text}, elapsed={elapsed_sec / 60.0:.1f} min"
    )


def make_job_id(variant: Variant, task: str, setting: Any, seed: int) -> str:
    """Create a stable, readable job ID."""
    return f"{variant.name}__{task}__{setting.run_slug}__s{seed}"


def hydra_string(value: str) -> str:
    """Quote a Hydra override string value."""
    return json.dumps(value, ensure_ascii=True)


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
