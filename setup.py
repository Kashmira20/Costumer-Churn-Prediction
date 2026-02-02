from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-end customer churn prediction system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/customer-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "plotly>=5.15.0",
        "streamlit>=1.25.0",
    ],
)