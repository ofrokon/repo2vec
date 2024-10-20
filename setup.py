from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repo2vec",
    version="0.1.0",
    author="Md Omar Faruk Rokon",
    author_email="mroko001@ucr.edu",
    description="A tool for creating vector representations of software repositories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ofrokon/repo2vec",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gensim>=4.0.0",
        "nltk>=3.5",
        "networkx>=2.5",
        "javalang>=0.13.0",
        "torch>=1.7.0",
        "torch-geometric>=2.0.0",
        "code2vec>=1.1.1",
    ],
)
