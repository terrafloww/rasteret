from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="rasteret",
    version="0.1.6",
    author="Sidharth Subramaniam",
    author_email="sid@terrafloww.com",
    description="Fast and efficient access to Cloud-Optimized GeoTIFFs (COGs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terrafloww/rasteret",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.10,<3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.23.2",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "ruff==0.8.6",
            "isort>=5.13.2",
            "flake8>=7.0.0",
            "httpx[http2]",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
