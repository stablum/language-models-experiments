"""Small function-step wrappers used by ClearML PipelineController."""


def train_tokenizer_stage_entry(**kwargs: object) -> str:
    from src.cli.output import timestamped_cli_output
    from src.cli.pipeline_steps import train_tokenizer_step

    with timestamped_cli_output():
        return train_tokenizer_step(**kwargs)


def train_model_stage_entry(**kwargs: object) -> str:
    from src.cli.output import timestamped_cli_output
    from src.cli.pipeline_steps import train_model_pipeline_step

    with timestamped_cli_output():
        return train_model_pipeline_step(**kwargs)


def evaluate_stage_entry(**kwargs: object) -> str:
    from src.cli.output import timestamped_cli_output
    from src.cli.pipeline_steps import evaluate_pipeline_step

    with timestamped_cli_output():
        return evaluate_pipeline_step(**kwargs)


def query_stage_entry(**kwargs: object) -> str:
    from src.cli.output import timestamped_cli_output
    from src.cli.pipeline_steps import query_pipeline_step

    with timestamped_cli_output():
        return query_pipeline_step(**kwargs)
