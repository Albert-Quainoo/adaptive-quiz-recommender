from importlib import import_module

PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "streamlit",
    "fastapi",
]


def main() -> None:
    for package_name in PACKAGES:
        try:
            module = import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version}")
        except Exception as exc:
            print(f"{package_name}: ERROR - {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
    except Exception as exc:
        print(f"GPU check failed: {exc}")


if __name__ == "__main__":
    main()
