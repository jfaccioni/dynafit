from setuptools import setup, find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    name="dynafit",
    version="0.2.0",
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
    install_requires=['pandas', 'PySide2', 'matplotlib', 'numpy', 'openpyxl', 'seaborn', 'scipy'],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'dynafit=src.interface:main',
        ],
    },
)
