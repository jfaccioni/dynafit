"""setup.py - setup file for DynaFit package."""

from setuptools import setup, find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="dynafit",
    version="0.3.0",
    author="Juliano Luiz Faccioni",
    author_email="julianofaccioni@gmail.com",
    description="A package to perform DynaFit analysis through a GUI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jfaccioni/dynafit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas >= 1.0',
                      'PySide2 >= 5.13',
                      'matplotlib >= 3.2',
                      'numpy >= 1.18',
                      'openpyxl >= 3.0',
                      'seaborn >= 0.10'],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'dynafit=src.interface:main',
        ],
    },
)
