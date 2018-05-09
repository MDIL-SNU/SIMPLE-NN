from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# required module
install_requires = [
    'numpy',
    'ase',
    'mpi4py',
    'pyyaml',
    'cffi',
    'tensorflow',
]

# TODO: extern C module add
setup(
    name='simple-nn',
    version='1.0.0',
    #description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Nanco-L/simple-nn', # temperary url 
    author='Kyuhyun Lee',
    author_email='khlee1992@naver.com',
    license='MIT',
    classifiers=[   # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # other arguments are listed here.

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

    ],
    #keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_data={'':['*.cpp', '*.h']},
    #project_urls={},
    python_requires='>=2.7, >=3, <4',
)