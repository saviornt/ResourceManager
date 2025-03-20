from setuptools import setup, find_packages

setup(
    name="ResourceManager",
    version="1.0.0",
    description="A resource manager that automatically assigns and handles CPU, memory, and GPU usage.",
    author="saviornt",
    project_url="https://github.com/saviornt/ResourceManager",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "psutil>=7.0.0",
        "pydantic>=2.10.6",
        "pynvml>=12.0.0",
        "numpy>=2.2.4",
        "numba>=0.61.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.12",
)
