from setuptools import setup, find_packages

setup(
    name="tradingview_analysis",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "alpaca-py",
        "pandas",
        "matplotlib",
        "python-dotenv",
        "numpy",
        "TA-Lib"
    ],
)
