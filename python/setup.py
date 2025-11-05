"""Setup script for the dynamical systems identification package."""

from setuptools import setup, find_packages

setup(
    name="sysid",
    version="0.1.0",
    description="RNN-based System Identification Package",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9,<3.12",  # Compatible with Intel Macs and modern clusters
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0,<2.0.0",  # NumPy 1.x for compatibility
        "scipy>=1.7.0",
        "pandas>=1.3.0",  # For CSV loading
        "mlflow>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
)
