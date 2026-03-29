from setuptools import setup, find_packages

setup(
    name="gpu-assembly-index",
    version="0.1.0",
    description="GPU-accelerated Molecular Assembly Index Calculator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tehlikeli107",
    url="https://github.com/Tehlikeli107/gpu-assembly-index",
    py_modules=["assembly_index", "gpu_engine", "nn_predictor", "validate", "pubchem_scan"],
    install_requires=[
        "numpy>=2.0",
        "rdkit>=2025.0",
        "torch>=2.0",
    ],
    extras_require={
        "cupy": ["cupy-cuda12x>=14.0"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
