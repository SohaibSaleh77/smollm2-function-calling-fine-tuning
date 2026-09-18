"""Setup file for the educational SmolLM2 function-calling project.

Install as an editable package from the project root:
    pip install -e .

`scripts/setup_env.py` wraps this (plus the torchao uninstall from the
original notebook). PyTorch is intentionally NOT listed here — install the
build matching your CUDA setup first (Colab ships a compatible one).
"""

from setuptools import find_packages, setup

setup(
    name="smollm2-glaive-fc",
    version="0.1.0",
    description="Educational example: QLoRA fine-tuning of SmolLM2-1.7B-Instruct on glaive-function-calling-v2.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.46",
        "peft>=0.13",
        "trl>=0.12",
        "datasets>=3.0",
        "accelerate>=1.0",
        "bitsandbytes>=0.44",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "tqdm>=4.66",
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
