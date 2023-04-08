from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="flyvision",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    author="Janne Lappalainen & Mason McGill",
    description="Library to build and analyze network simulations of the Drosophila visual system.",
)
