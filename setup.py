from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES


for scheme in INSTALL_SCHEMES.values():
    scheme["data"] = scheme["purelib"]

setup(
    name="cluster-mlp",
    version="0.2",
    description="Module for cluster genetic algorithm",
    url="https://github.com/ulissigroup/cluster-mlp",
    author="Saurabh Sivakumar, Rajesh Raju, Zachary Ulissi",
    author_email="zulissi@andrew.cmu.edu",
    packages=find_packages(),
    include_package_data=False,
    install_requires=["ase", "deap", "dask"],
    long_description="""Module for implementing cluster GA. Future integration with MLP""",
)
