import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
with open("requirements.txt") as file:
    requirements_txt = file.read().splitlines()


setuptools.setup(
    name="ptk-patrickctrf",
    version="0.0.1",
    author="patrickctrf",
    author_email="patrickctrf@gmail.com",
    description="Package for Python programming by Patrick Ferreira",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patrickctrf/ptk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements_txt
)
