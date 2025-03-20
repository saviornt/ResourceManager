from setuptools import setup, find_packages

setup(
    name="ResourceManager",
    version="1.0.0",
    description="A resource manager that automatically assigns and handles CPU, memory, and GPU usage.",
    author="saviornt",
    project_urls={
        "Homepage": "https://github.com/saviornt/ResourceManager",
        "Bug Tracker": "https://github.com/saviornt/ResourceManager/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "annotated-types>=0.7.0",
        "fastapi>=0.115.11",
        "httpx>=0.28.1",
        "llvmlite>=0.44.0",
        "numba>=0.61.0",
        "numpy>=2.1.3",
        "psutil>=7.0.0",
        "pydantic>=2.10.6",
        "pydantic_core>=2.27.2",
        "pynvml>=12.0.0",
        "python-dotenv>=1.0.1",
        "typing_extensions>=4.12.2",
        "uvicorn>=0.34.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.5",
            "pytest-asyncio>=0.25.3",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: System :: Monitoring",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
    ],
    python_requires=">=3.12",
)
