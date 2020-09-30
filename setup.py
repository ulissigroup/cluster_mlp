from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES


for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

setup(name='cluster_GA',
      version='0.1',
      description='Module for cluster genetic algorithm',
      url='https://github.com/ulissigroup/cluster-mlp',
      author='Saurabh Sivakumar, Rajesh Raju, Zachary Ulissi',
      author_email='zulissi@andrew.cmu.edu',
      license='GPL',
      platforms=[],
      packages=find_packages(),
      scripts=[],
      include_package_data=False,
      install_requires=['ase>=3.19.1',
			'numpy',
			'matplotlib',
            'amptorch',
			'deap'],
      long_description='''Module for implementing cluster GA. Future integration with MLP''',)
