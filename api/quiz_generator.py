import torch

from api.schemas import QuizGenerationRequest, QuizResponse
from api.model_loader import load_model, load_tokenizer
from api.prompt_builder import build_quiz_messages
from api.response_parser import parse_quiz_messages

def token_budget(question_count: int) -> int:
    return 200 + 400 * question_count


def generate_raw_quiz(request: QuizGenerationRequest, max_new_tokens: int | None = None) -> str:
    if max_new_tokens is None:
        max_new_tokens = token_budget(request.question_count)

    tokenizer = load_tokenizer()
    model = load_model()

    messages =  build_quiz_messages(request)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = output[0, inputs["input_ids"].shape[1] :]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens = True
    )
    return response


def generate_quiz(request: QuizGenerationRequest, max_new_tokens: int | None = None,) -> QuizResponse:
    raw_response = generate_raw_quiz(request, max_new_tokens)

    quiz = parse_quiz_messages(raw_response)

    if len(quiz.questions) != request.question_count:
        raise ValueError(
            f"Expected {request.question_count} questions, "
            f"but received {len(quiz.questions)} questions."
        )

    return quiz




