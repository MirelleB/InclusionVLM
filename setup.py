from setuptools import setup, find_packages

setup(
    name="americasnlp-repo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers>=4.45.0",
        "accelerate",
        "peft",
        "datasets",
        "evaluate",
        "sacrebleu",
        "trl",
        "bitsandbytes",
        "pandas",
        "tqdm",
        "pillow",
    ],
    author="AmericasNLP Contributors",
    description="Multimodal training and evaluation for low-resource languages.",
    python_requires=">=3.10",
)
