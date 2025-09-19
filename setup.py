from setuptools import setup, find_packages

setup(
    name="deladect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.18.5",
        "scipy==1.6.0",
        "pandas==1.3.5",  # Add a specific version for pandas too
        "matplotlib==3.3.4",
        "scikit-image==0.18.1",
        "Pillow==8.2.0",   # Add a specific version for Pillow
        "crackdect", # If crackdect has a specific version
    ],
    entry_points={
        "console_scripts": [
            "shift_correction=aux_scripts.shift_correction_code.shift_correction:main",
        ],
    },
    author="Vasco D.C Pires, Matthias Rettl",
    description="A package for crack and delamination detection with shift correction.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo-url",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
