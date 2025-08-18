"""
Setup script for Boston Housing ML Project
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Boston Housing Machine Learning Project"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="boston-housing-ml",
    version="1.0.0",
    description="Machine Learning project for predicting Boston housing prices",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/boston_housing_ml_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "boston-housing-demo=boston_housing_ml_project.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning, regression, housing, boston, scikit-learn, data-science",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/boston_housing_ml_project/issues",
        "Source": "https://github.com/yourusername/boston_housing_ml_project",
        "Documentation": "https://github.com/yourusername/boston_housing_ml_project#readme",
    },
)
