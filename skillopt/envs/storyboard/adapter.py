"""Storyboard environment adapter for SkillOpt."""
from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.storyboard.dataloader import StoryboardDataLoader
from skillopt.envs.storyboard.rollout import run_batch
from skillopt.envs.storyboard.pipeline import (
    compose_skill,
    DEFAULT_MAX_TOKENS,
)
from skillopt.gradient.reflect import run_minibatch_reflect


class StoryboardAdapter(EnvAdapter):
    """Environment adapter for script-to-shots storyboard generation."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "5:1:2",
        split_seed: int = 42,
        split_output_dir: str = "",
        exec_timeout: int = 300,
        workers: int = 2,
        analyst_workers: int = 2,
        failure_only: bool = False,
        minibatch_size: int = 2,
        edit_budget: int = 3,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 16384,
        pipeline_mode: str = "multi_agent",
        skill_base_dir: str = "",
        max_tokens_per_stage: dict | None = None,
    ) -> None:
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.pipeline_mode = pipeline_mode
        self.skill_base_dir = skill_base_dir
        self.max_tokens_per_stage = max_tokens_per_stage or DEFAULT_MAX_TOKENS
        self.frozen_skills = self._load_frozen_skills(skill_base_dir)
        self.dataloader = StoryboardDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    @staticmethod
    def _load_frozen_skills(skill_base_dir: str) -> dict[str, str]:
        """Load Editor/Continuity/QA skills (frozen, not optimized)."""
        frozen = {}
        for name in ("editor", "continuity", "qa"):
            path = os.path.join(skill_base_dir, name, "SKILL.md")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    frozen[name] = f.read()
            else:
                frozen[name] = ""
        return frozen

    @staticmethod
    def compose_initial_skill(skill_base_dir: str) -> str:
        """Compose Director+DP into a single optimizable skill string."""
        director_path = os.path.join(skill_base_dir, "director", "SKILL.md")
        dp_path = os.path.join(skill_base_dir, "dp", "SKILL.md")
        director = ""
        dp = ""
        if os.path.exists(director_path):
            with open(director_path, encoding="utf-8") as f:
                director = f.read()
        if os.path.exists(dp_path):
            with open(dp_path, encoding="utf-8") as f:
                dp = f.read()
        return compose_skill(director, dp)

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        if cfg.get("skill_init") == "auto_compose" and self.skill_base_dir:
            composite = self.compose_initial_skill(self.skill_base_dir)
            out_root = cfg.get("out_root", ".")
            os.makedirs(out_root, exist_ok=True)
            composite_path = os.path.join(out_root, "composite_skill_init.md")
            with open(composite_path, "w", encoding="utf-8") as f:
                f.write(composite)
            cfg["skill_init"] = composite_path
            print(f"  [pipeline] auto-composed skill from {self.skill_base_dir} ({len(composite)} chars)")

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
            task_timeout=self.exec_timeout + 120,
            frozen_skills=self.frozen_skills,
            max_tokens_per_stage=self.max_tokens_per_stage,
            pipeline_mode=self.pipeline_mode,
        )

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        patches_dir = kwargs.get("patches_dir", os.path.join(out_dir, "patches"))
        return run_minibatch_reflect(
            results=results,
            skill_content=skill_content,
            prediction_dir=prediction_dir,
            patches_dir=patches_dir,
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=kwargs.get("random_seed"),
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=kwargs.get("step_buffer_context", ""),
            meta_skill_context=kwargs.get("meta_skill_context", ""),
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def get_task_types(self) -> list[str]:
        return ["storyboard"]
