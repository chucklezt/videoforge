from setuptools import setup, find_packages

setup(
    name="videoforge",
    version="0.1.0",
    description="Video generation training pipeline for AMD ROCm",
    author="chuck",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    entry_points={
        "console_scripts": [
            "videoforge=videoforge.__main__:main",
        ],
    },
)
