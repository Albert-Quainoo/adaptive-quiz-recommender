import torch

from api.model_loader import load_model, load_tokenizer


def main() -> None:
    tokenizer = load_tokenizer()
    model = load_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an educational quiz generator. "
                "Return one concise multiple-choice question."
            ),
        },
        {
            "role": "user",
            "content": "Generate one introductory question about queues.",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, inputs["input_ids"].shape[1] :]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\nGenerated response:\n")
    print(response)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"\nGPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")


if __name__ == "__main__":
    main()
