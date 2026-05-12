"""Config default resolution for the model-training CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import click

from src.ml_core import cfg as core_cfg
from src.ml_core.cli import config as cli_config


MODEL_TRAINING_CONFIG_SECTION = "model-training"
OPTUNA_CONFIG_SECTION = "optuna"
TRAIN_CONFIG_SECTION = "train"
EVALUATE_CONFIG_SECTION = "evaluate"
QUERY_CONFIG_SECTION = "query"
EXPLICIT_PARAMETER_SOURCES = {"COMMANDLINE", "ENVIRONMENT"}
SHARED_DATA_PARAMETERS = (
    "corpus",
    "dataset_id",
    "source_split",
    "train_ratio",
    "split_seed",
    "text_column",
    "streaming",
)
TRAIN_DEFAULT_KEY_MAP = {
    "tokenizer_model_name": "tokenizer_model_name",
    "smoothing": "smoothing",
    "unigram_weight": "unigram_weight",
    "bigram_weight": "bigram_weight",
    "trigram_weight": "trigram_weight",
    "discount": "discount",
    "limit": "training_limit",
    "text_normalization": "text_normalization",
}
EVALUATE_DEFAULT_KEY_MAP = {
    "evaluation_partition": "evaluation_partition",
    "top_k": "top_k",
    "limit": "evaluation_limit",
}
QUERY_DEFAULT_KEY_MAP = {
    "prompt": "query_prompt",
    "max_tokens": "query_max_tokens",
    "top_k": "query_top_k",
    "decoding": "query_decoding",
    "temperature": "query_temperature",
    "seed": "query_seed",
}


class StageDefaults(core_cfg.BaseCfg):
    """Config defaults loaded for the model-training stage sections."""

    pipeline: Mapping[str, object]
    train: Mapping[str, object]
    evaluate: Mapping[str, object]
    query: Mapping[str, object]

    @classmethod
    def load(cls) -> "StageDefaults":
        return cls(
            pipeline=cli_config.load_defaults_from_sections(
                (MODEL_TRAINING_CONFIG_SECTION,)
            ),
            train=cli_config.load_defaults_from_sections((TRAIN_CONFIG_SECTION,)),
            evaluate=cli_config.load_defaults_from_sections((EVALUATE_CONFIG_SECTION,)),
            query=cli_config.load_defaults_from_sections((QUERY_CONFIG_SECTION,)),
        )

    def resolve_stage(
        self,
        ctx: click.Context | None,
        *,
        parameter_name: str,
        current_value: object,
        stage: str,
        stage_key: str | None = None,
    ) -> object:
        return resolve_stage_default(
            ctx,
            parameter_name=parameter_name,
            current_value=current_value,
            pipeline_defaults=self.pipeline,
            stage_defaults=self.for_stage(stage),
            stage_key=stage_key,
        )

    def resolve_shared(
        self,
        ctx: click.Context | None,
        *,
        parameter_name: str,
        current_value: object,
        stages: tuple[str, ...],
    ) -> object:
        return resolve_consistent_stage_default(
            ctx,
            parameter_name=parameter_name,
            current_value=current_value,
            pipeline_defaults=self.pipeline,
            candidates=tuple(
                (stage, self.for_stage(stage), parameter_name)
                for stage in stages
            ),
        )

    def resolve_limit(
        self,
        ctx: click.Context | None,
        *,
        parameter_name: str,
        current_value: int | None,
        stage: str,
        global_limit: int | None,
    ) -> int | None:
        return resolve_stage_limit(
            ctx,
            parameter_name=parameter_name,
            current_value=current_value,
            pipeline_defaults=self.pipeline,
            stage_defaults=self.for_stage(stage),
            global_limit=global_limit,
        )

    def for_stage(self, stage: str) -> Mapping[str, object]:
        if stage == "train":
            return self.train
        if stage == "evaluate":
            return self.evaluate
        if stage == "query":
            return self.query
        raise KeyError(stage)


def resolve_stage_default(
    ctx: click.Context | None,
    *,
    parameter_name: str,
    current_value: object,
    pipeline_defaults: Mapping[str, object],
    stage_defaults: Mapping[str, object],
    stage_key: str | None = None,
) -> object:
    if parameter_is_explicit(ctx, parameter_name):
        return current_value
    if parameter_name in pipeline_defaults:
        return current_value

    resolved_stage_key = stage_key or parameter_name
    if resolved_stage_key in stage_defaults:
        return stage_defaults[resolved_stage_key]
    return current_value


def resolve_consistent_stage_default(
    ctx: click.Context | None,
    *,
    parameter_name: str,
    current_value: object,
    pipeline_defaults: Mapping[str, object],
    candidates: tuple[tuple[str, Mapping[str, object], str], ...],
) -> object:
    if parameter_is_explicit(ctx, parameter_name):
        return current_value
    if parameter_name in pipeline_defaults:
        return current_value

    values = _stage_config_values(candidates)
    if not values:
        return current_value
    return _consistent_stage_value(parameter_name, values)


def resolve_stage_limit(
    ctx: click.Context | None,
    *,
    parameter_name: str,
    current_value: int | None,
    pipeline_defaults: Mapping[str, object],
    stage_defaults: Mapping[str, object],
    global_limit: int | None,
) -> int | None:
    if parameter_is_explicit(ctx, parameter_name):
        return current_value
    if parameter_name in pipeline_defaults:
        return current_value
    if parameter_is_explicit(ctx, "limit") or "limit" in pipeline_defaults:
        return global_limit
    if "limit" in stage_defaults:
        return _optional_int_default(stage_defaults["limit"], name="limit")
    return global_limit


def parameter_is_explicit(ctx: click.Context | None, parameter_name: str) -> bool:
    if ctx is None or parameter_name not in ctx.params:
        return False
    source = ctx.get_parameter_source(parameter_name)
    return getattr(source, "name", None) in EXPLICIT_PARAMETER_SOURCES


def load_model_training_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = cli_config.load_defaults_from_sections(("defaults", "clearml"))
    stage_defaults = StageDefaults.load()
    optuna_defaults = cli_config.load_defaults_from_sections((OPTUNA_CONFIG_SECTION,))

    defaults.update(
        _consistent_config_values(
            current_defaults=defaults,
            candidates=(
                ("train", stage_defaults.train),
                ("evaluate", stage_defaults.evaluate),
                ("query", stage_defaults.query),
            ),
            parameter_names=SHARED_DATA_PARAMETERS,
        )
    )
    defaults.update(
        _consistent_config_values(
            current_defaults=defaults,
            candidates=(
                ("train", stage_defaults.train),
                ("evaluate", stage_defaults.evaluate),
                ("query", stage_defaults.query),
            ),
            parameter_names=("model_name",),
        )
    )
    defaults.update(
        _mapped_config_values(
            stage_defaults.train,
            TRAIN_DEFAULT_KEY_MAP,
        )
    )
    defaults.update(
        _mapped_config_values(
            stage_defaults.evaluate,
            EVALUATE_DEFAULT_KEY_MAP,
        )
    )
    defaults.update(
        _mapped_config_values(
            stage_defaults.query,
            QUERY_DEFAULT_KEY_MAP,
        )
    )
    defaults.update(optuna_defaults)
    defaults.update(stage_defaults.pipeline)
    return defaults


def _stage_config_values(
    candidates: tuple[tuple[str, Mapping[str, object], str], ...],
) -> list[tuple[str, object]]:
    return [
        (section, defaults[key])
        for section, defaults, key in candidates
        if key in defaults
    ]


def _consistent_stage_value(
    parameter_name: str,
    values: Sequence[tuple[str, object]],
) -> object:
    first_value = values[0][1]
    conflicts = [
        (section, value)
        for section, value in values[1:]
        if value != first_value
    ]
    if not conflicts:
        return first_value

    formatted_values = ", ".join(
        f"[{section}] {parameter_name}={value!r}"
        for section, value in values
    )
    raise click.ClickException(
        f"Conflicting pipeline defaults for {parameter_name!r}: {formatted_values}. "
        "Set one shared value in [defaults] or [model-training], "
        "or make the stage sections match."
    )


def _optional_int_default(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise click.ClickException(f"Config default {name!r} must be an integer.")
    return value


def _consistent_config_values(
    *,
    current_defaults: Mapping[str, object],
    candidates: tuple[tuple[str, Mapping[str, object]], ...],
    parameter_names: tuple[str, ...],
) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for parameter_name in parameter_names:
        values = [
            (section, defaults[parameter_name])
            for section, defaults in candidates
            if parameter_name in defaults
        ]
        if not values:
            continue
        first_value = _consistent_stage_value(parameter_name, values)
        if (
            parameter_name not in current_defaults
            or first_value != current_defaults[parameter_name]
        ):
            resolved[parameter_name] = first_value
    return resolved


def _mapped_config_values(
    defaults: Mapping[str, object],
    key_map: Mapping[str, str],
) -> dict[str, object]:
    return {
        target_key: defaults[source_key]
        for source_key, target_key in key_map.items()
        if source_key in defaults
    }
