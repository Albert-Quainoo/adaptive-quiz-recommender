import torch

from api.schemas import QuizGenerationRequest
from api.model_loader import load_model, load_tokenizer
from api.prompt_builder import build_quiz_messages

def generate_raw_quiz(request: QuizGenerationRequest, max_new_tokens: int = 1200) -> str:
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





