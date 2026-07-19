from api.model_loader import get_model_id, load_tokenizer


def main() -> None:
    tokenizer = load_tokenizer()

    messages = [
        {
            "role": "system",
            "content": "You generate clear educational quiz questions.",
        },
        {
            "role": "user",
            "content": "Create one multiple-choice question about stacks.",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(prompt, return_tensors="pt")

    print("Model:", get_model_id())
    print("Tokenizer:", type(tokenizer).__name__)
    print("Token count:", encoded.input_ids.shape[1])
    print("\nPrompt preview:\n")
    print(prompt)


if __name__ == "__main__":
    main()
