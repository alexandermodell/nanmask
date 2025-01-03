from setuptools import setup, find_packages

setup(
    name="nanmask",
    version="0.1.0",
    description="A package for handling NaN masking and unmasking in NumPy arrays",
    author="Alexander Modell",
    packages=find_packages(),
    install_requires=["numpy"],
)
