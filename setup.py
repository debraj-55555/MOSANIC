from setuptools import setup, find_packages
setup(
    name="mosanic-ccc",
    version="1.0.0",
    packages=find_packages(),
    entry_points={"console_scripts": ["mosanic=mosanic.cli:main"]},
)
