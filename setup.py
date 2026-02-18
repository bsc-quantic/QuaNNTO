from setuptools import setup, find_packages

setup(
    name="quannto",
    version="0.1.0",
    description="QuaNNTO: Exact simulator of CV Quantum Optical Neural Networks",
    packages=find_packages(),  # discovers quannto and subpackages
    include_package_data=True,
    python_requires=">=3.10",
)