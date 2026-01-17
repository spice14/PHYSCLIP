"""
PHYSCLIP: Physics-informed Contrastive Learning for Interpretable Representation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="physclip",
    version="0.1.0",
    author="PHYSCLIP Contributors",
    author_email="",
    description="Physics-informed Contrastive Learning for Interpretable Representation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spice14/PHYSCLIP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
