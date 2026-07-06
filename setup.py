from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="diabetes-risk-prediction",
        version="2.0.0",
        description="Production-grade MLOps Diabetes Risk Prediction System",
        author="Rohitt Dusane",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        python_requires=">=3.10",
        install_requires=[
            "numpy>=1.24",
            "pandas>=2.0",
            "scikit-learn>=1.3",
            "fastapi>=0.110",
            "uvicorn>=0.27",
            "joblib>=1.3",
            "pydantic>=2.0",
            "pyyaml>=6.0"
        ],
    )