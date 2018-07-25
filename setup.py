from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# required module
# TODO: version check
install_requires = [
    'numpy',
    'ase>=3.10.0',
    'pyyaml>=3.10',
    'cffi>=1.0.0',
    'psutil'
]

# C extension library for calculating symmetry function
extension_libsymf = Extension(
    'simple_nn.features.symmetry_function.libsymf', 
    sources=[
        'simple_nn/features/symmetry_function/calculate_sf.cpp',
        'simple_nn/features/symmetry_function/symmetry_functions.cpp'
        ],
    languages='C++',
    extra_compile_args=[
        '-O3',
        ]
    )

# C extension library for calculating gdf
extension_libgdf = Extension(
    'simple_nn.utils.libgdf',
    sources = ['simple_nn/utils/gdf.cpp'],
    language='C++',
    extra_compile_args=[
        '-O3'
        ]
    )

# TODO: extern C module add
# TODO: install requires add
# FIXME: fill the empty part
setup(
    name='simple-nn',
    version='0.2.6',
    description='Package for generating atomic potentials using neural network.',
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
        'Development Status :: 3 - Alpha',

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
    python_requires='>=2.7, <4',
    install_requires=install_requires,
    ext_modules=[
        extension_libsymf,
        extension_libgdf
        ]
)
