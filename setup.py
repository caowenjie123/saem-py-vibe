from setuptools import setup, find_packages

setup(
    name="saemix",
    version="0.1.0",
    description="Stochastic Approximation Expectation Maximization (SAEM) Algorithm for Python",
    author="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.7",
)