from setuptools import setup, find_packages

setup(
    name="neuroscience_sleep_scoring",
    version="1.0.0",
    description="A package for sleep scoring rodent EEG+EMG.",
    author="Lizzie Tilden",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/neuroscience_sleep_scoring",
    packages=find_packages(include=["neuroscience_sleep_scoring", "neuroscience_sleep_scoring.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "psutil",
        "pandas",
        "natsort",
        "pyedflib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)