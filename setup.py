import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mfgpflow",
    version="0.0.0",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    package_dir={"": "."},  # Explicitly tell setuptools that packages are inside 'src/'
    python_requires=">=3.12",
    install_requires=[
    "numpy",
    "h5py",
    ],
)
