import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
    name = "pyfoma",
    version = "0.0.10",
    author = "Mans Hulden",
    author_email = "mans.hulden@colorado.edu",
    description = "Python Finite-State Toolkit",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/mhulden/pyfoma",
    project_urls = {
        "Bug Tracker": "https://github.com/mhulden/pyfoma/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    py_modules = ["pyfoma"],
    python_requires = ">=3.6",
    install_requires = [
        "graphviz<0.16", "IPython"
    ]
)
