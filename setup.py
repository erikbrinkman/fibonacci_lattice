import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fiblat",
    version="0.0.1",
    author="Erik Brinkman",
    author_email="erik.brinkman@gmail.com",
    description="A package for generating evenly distributed points on a sphere",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erikbrinkman/fibonacci_lattice",
    packages=setuptools.find_packages(),
    classifiers=[],
    python_requires=">=3.6",
)
