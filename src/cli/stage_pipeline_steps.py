"""Small function-step wrappers used by ClearML PipelineController."""


def train_tokenizer_stage_entry(**kwargs: object) -> str:
    from src.cli.pipeline_steps import train_tokenizer_pipeline_step

    return train_tokenizer_pipeline_step(**kwargs)


def train_model_stage_entry(**kwargs: object) -> str:
    from src.cli.pipeline_steps import train_model_pipeline_step

    return train_model_pipeline_step(**kwargs)


def evaluate_stage_entry(**kwargs: object) -> str:
    from src.cli.pipeline_steps import evaluate_pipeline_step

    return evaluate_pipeline_step(**kwargs)


def query_stage_entry(**kwargs: object) -> str:
    from src.cli.pipeline_steps import query_pipeline_step

    return query_pipeline_step(**kwargs)
