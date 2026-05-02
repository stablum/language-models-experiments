"""Click command helpers for repo-level TOML defaults."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import click

from src.cli.output import timestamped_cli_output


CONFIG_ENVVAR = "LME_CONFIG_FILE"
DEFAULT_CONFIG_PATH = Path("config.toml")
SHARED_SECTIONS = ("defaults", "clearml")
KEY_ALIASES = {
    "model": "model_name",
}


class ConfigurableCommand(click.Command):
    """Click command that loads omitted option defaults from config.toml."""

    def __init__(
        self,
        *args: Any,
        config_section: str,
        default_loader: Callable[[str], dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config_section = config_section
        self.default_loader = default_loader or load_command_defaults

    def make_context(
        self,
        info_name: str | None,
        args: list[str],
        parent: click.Context | None = None,
        **extra: Any,
    ) -> click.Context:
        for key, value in self.context_settings.items():
            if key not in extra:
                extra[key] = value

        config_defaults = self.default_loader(self.config_section)
        if config_defaults:
            valid_parameter_names = {
                parameter.name
                for parameter in self.params
                if parameter.name is not None
            }
            default_map = dict(extra.get("default_map") or {})
            default_map.update(
                {
                    key: value
                    for key, value in config_defaults.items()
                    if key in valid_parameter_names
                }
            )
            extra["default_map"] = default_map

        ctx = self.context_class(self, info_name=info_name, parent=parent, **extra)
        with ctx.scope(cleanup=False):
            self.parse_args(ctx, args)
        return ctx

    def main(self, *args: Any, **kwargs: Any) -> Any:
        with timestamped_cli_output():
            return super().main(*args, **kwargs)


def configured_command(
    config_section: str,
    default_loader: Callable[[str], dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    return click.command(
        cls=ConfigurableCommand,
        config_section=config_section,
        default_loader=default_loader,
        **kwargs,
    )


def load_command_defaults(config_section: str) -> dict[str, Any]:
    return load_defaults_from_sections((*SHARED_SECTIONS, config_section))


def load_defaults_from_sections(sections: tuple[str, ...]) -> dict[str, Any]:
    config_path = configured_path()
    if not config_path.exists():
        if os.environ.get(CONFIG_ENVVAR):
            raise click.ClickException(
                f"Config file from {CONFIG_ENVVAR} does not exist: {config_path}"
            )
        return {}

    data = load_config(config_path)
    defaults: dict[str, Any] = {}
    for section in sections:
        defaults.update(section_defaults(data, section))
    return defaults


def configured_path() -> Path:
    env_path = os.environ.get(CONFIG_ENVVAR)
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(config_path: Path) -> Mapping[str, Any]:
    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise click.ClickException(f"Invalid TOML in {config_path}: {error}") from error

    if not isinstance(data, Mapping):
        raise click.ClickException(f"Config file must contain TOML tables: {config_path}")
    return data


def section_defaults(data: Mapping[str, Any], section: str) -> dict[str, Any]:
    section_data = first_mapping(data, section, section.replace("_", "-"))
    if section_data is None:
        return {}
    return normalize_keys(section_data)


def first_mapping(data: Mapping[str, Any], *section_names: str) -> Mapping[str, Any] | None:
    for section_name in section_names:
        value = data.get(section_name)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise click.ClickException(f"Config section [{section_name}] must be a TOML table.")
        return value
    return None


def normalize_keys(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        normalize_key(key): value
        for key, value in values.items()
    }


def normalize_key(key: str) -> str:
    normalized = key.replace("-", "_")
    return KEY_ALIASES.get(normalized, normalized)
