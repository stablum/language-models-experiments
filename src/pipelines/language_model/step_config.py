"""Shared ClearML function-step setup for language-model pipeline definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.ml_core import pipeline as core_pipeline
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import monitors as lm_monitors


@dataclass(frozen=True)
class StepCfg:
    project_name: str
    output_uri: str | None
    tags: tuple[str, ...]
    config_file: Path | None
    queue: str | None

    def add(
        self,
        pipeline: object,
        *,
        name: str,
        function: object,
        function_kwargs: Mapping[str, object],
        task_type: str,
        parents: Sequence[str] = (),
    ) -> None:
        add_function_step = getattr(pipeline, "add_function_step")
        step_kwargs = {
            "name": name,
            "function": function,
            "function_kwargs": {
                **function_kwargs,
                **self.function_kwargs(),
            },
            "task_name": name,
            "task_type": task_type,
            "monitor_artifacts": lm_monitors.pipeline_artifact_monitors()[name],
            "monitor_metrics": lm_monitors.pipeline_metric_monitors()[name],
            "stage": name,
            **self.step_options(),
        }
        if parents:
            step_kwargs["parents"] = list(parents)
        add_function_step(**step_kwargs)

    def function_kwargs(self) -> dict[str, object]:
        return {
            "clearml_output_uri": self.output_uri,
            "clearml_tags": "\n".join(self.tags),
            "clearml_config_file": str(self.config_file) if self.config_file else None,
        }

    def step_options(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "execution_queue": self.queue,
            "output_uri": core_pipeline.output_uri_value(self.output_uri),
            "auto_connect_frameworks": False,
            "auto_connect_arg_parser": False,
            "pre_execute_callback": lm_def.stage_gate_callback,
            "tags": list(self.tags) if self.tags else None,
        }
