"""CLI for reproducible, review-first grounded question generation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from api.prompt_builder import PROMPT_VERSION
from authoring.grounded_batch import (
    BatchConfig,
    BatchGenerationError,
    GenerationValue,
    current_git_commit,
    generate_batch,
)
from taxonomy.loader import course_paths, course_provenance_path


class TransformersBatchModel:
    """The existing local Transformers model behind the batch protocol."""

    def __init__(self, model_id: str):
        from api.model_loader import (
            get_model_id,
            load_model,
            load_tokenizer,
        )

        configured = get_model_id()
        if configured != model_id:
            raise BatchGenerationError(
                f"--model-id {model_id} does not match MODEL_REPOSITORY {configured}"
            )

        self.model_id = model_id
        self.tokenizer = load_tokenizer()
        self.model = load_model()
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or self.tokenizer.init_kwargs.get("_commit_hash")
        )
        if not self.model_revision:
            raise BatchGenerationError(
                "the loaded model exposes no immutable model revision"
            )

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int,
        generation_parameters: dict[str, GenerationValue],
    ) -> str:
        import torch
        from transformers import set_seed

        set_seed(seed)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        parameters = dict(generation_parameters)
        parameters["pad_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            output = self.model.generate(**inputs, **parameters)

        generated = output[0, inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(generated, skip_special_tokens=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a grounded batch into a pending review queue."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--skill-id", action="append", required=True)
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--questions-per-skill", type=int)
    inventory.add_argument(
        "--all-blueprint-intents",
        action="store_true",
        help="generate each reviewed blueprint intent exactly once",
    )
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--max-attempts-per-question", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing incomplete batch without regenerating accepted slots.",
    )
    parser.add_argument(
        "--allow-intent-reuse",
        action="store_true",
        help="Explicitly permit cycling reviewed intents within one skill batch.",
    )
    parser.add_argument(
        "--difficulty",
        choices=("introductory", "intermediate", "advanced", "mixed"),
        default="intermediate",
    )
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    if arguments.prompt_version != PROMPT_VERSION:
        print(
            f"--prompt-version must be {PROMPT_VERSION} for the current prompt",
            file=sys.stderr,
        )
        return 2

    try:
        config = BatchConfig(
            batch_id=arguments.batch_id,
            skill_ids=arguments.skill_id,
            questions_per_skill=arguments.questions_per_skill,
            base_seed=arguments.base_seed,
            model_id=arguments.model_id,
            prompt_version=arguments.prompt_version,
            difficulty=arguments.difficulty,
            max_attempts_per_question=arguments.max_attempts_per_question,
            allow_intent_reuse=arguments.allow_intent_reuse,
            generation_parameters={
                "max_new_tokens": arguments.max_new_tokens,
                "do_sample": True,
                "temperature": arguments.temperature,
                "top_p": arguments.top_p,
            },
        )
        git_commit = current_git_commit()
        model = TransformersBatchModel(arguments.model_id)
        skills_path, references_path = course_paths("ai")
        result = generate_batch(
            config,
            model,
            arguments.output,
            skills_path=skills_path,
            references_path=references_path,
            provenance_path=course_provenance_path("ai"),
            git_commit=git_commit,
            resume=arguments.resume,
        )
    except (BatchGenerationError, ValidationError, ValueError) as error:
        print(f"Batch generation failed: {error}", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2, exclude={"questions", "attempts"}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
