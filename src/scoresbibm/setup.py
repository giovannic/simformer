from setuptools import find_packages, setup
import os

NAME = "scoresbibm"
VERSION = "0.0.1"
DESCRIPTION = "Score-based inference benchmark"
AUTHOR = "Anonymous"

entry_points={
    'console_scripts': [
        'scoresbi = scoresbibm.scripts:main',
    ],
}



REQUIRED = [
    "numpy",
    "matplotlib",
    "jax",
    "torch>=2.6",
    "torchaudio",
    "torchvision",
    "hydra-core",
    "hydra-submitit-launcher",
    "hydra-optuna-sweeper",
    "omegaconf",
    # sbi is installed with --no-deps (see post-install below)
    # because sbi<=0.25 caps torch<2.6 which has no Python 3.14 wheels
    "pyro-ppl",
    "pyknos>=0.16.0",
    "scikit-learn",
    "scipy",
    "tensorboard",
    "arviz",
    "joblib>=1.0.0",
    "pillow",
    "zuko>=1.2.0",
    "optuna",
    "tueplots",
    "seaborn",
    "pandas",
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    package_dir={"scoresbibm": "scoresbibm"},
    install_requires=REQUIRED,
    entry_points=entry_points,
)

# When using pip (not uv), install sbi and sbibm with --no-deps
# to avoid torch<2.6 pin conflict on Python 3.14.
# With uv, run manually: uv pip install sbi sbibm --no-deps
os.system("pip install sbi --no-deps -q")
os.system("pip install sbibm --no-deps -q")

