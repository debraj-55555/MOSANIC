from setuptools import setup, find_packages
setup(
    name="mosaic-ccc",
    version="1.0.0",
    packages=find_packages(),
    entry_points={"console_scripts": ["mosaic=mosaic.cli:main"]},
)
