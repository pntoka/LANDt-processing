from setuptools import setup, find_packages

setup(
    name='landt_processing',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'tqdm', 'openpyxl'],  # Add any dependencies here
    tests_require=['pytest'],
    author='Piotr Toka',
    author_email='pnt17@ic.ac.uk',
    description='A package for procressing and analysing data from LANDt files',
    url='https://github.com/pntoka/LANDt-processing.git',
)
