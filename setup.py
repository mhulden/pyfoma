import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
	long_description = fh.read()

github_permalink = "https://raw.githubusercontent.com/mhulden/pyfoma/main/docs/images/"
fixed_readme = long_description.replace("./docs/images/", github_permalink)

setuptools.setup(
	name = "pyfoma",
	version = "v1.0.2",
	author = "Mans Hulden",
	author_email = "mans.hulden@colorado.edu",
	description = "Python Finite-State Toolkit",
	long_description = fixed_readme,
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
	package_dir = {"": "src"},
	packages = setuptools.find_packages(where="src"),
	python_requires = ">=3.6",
	install_requires = [
		"IPython", "graphviz"
	]
)
