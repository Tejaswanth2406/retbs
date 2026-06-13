from setuptools import setup, find_packages

setup(
    name="retbs",
    version="1.0.0",
    description="RETBS: Radiate→Evolve→Transform→Blend→Sustain — Evolutionary Fine-Tuning Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="RETBS Research",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "trl>=0.8",
        "datasets>=2.18",
        "accelerate>=0.29",
        "deepspeed>=0.14",
        "ray>=2.10",
        "numpy>=1.26",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "rich>=13.7",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "black", "ruff"],
        "axolotl": ["axolotl>=0.4"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "retbs-run=retbs.examples.quickstart:main",
        ]
    },
)
