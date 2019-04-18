#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit thesis_analysis/__version__.py
version = {}
with open(os.path.join(here, 'thesis_analysis', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='thesis_analysis',
    version=version['__version__'],
    description="Analysis scripts used during my PhD thesis, mainly for interactive analysis of Barcode output.",
    long_description=readme + '\n\n',
    author="E. G. Patrick Bos",
    author_email='egpbos@gmail.com',
    url='https://github.com/egpbos/thesis_analysis',
    packages=[
        'thesis_analysis',
    ],
    py_modules=[
        'barcode_interactive_analysis',
        'barcode_plog_analysis'
    ],
    include_package_data=True,
    license="MIT license",
    zip_safe=False,
    keywords='thesis_analysis',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'egp',
        'pandas',
    ],  # FIXME: add your package's dependencies to this list
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'dev':  ['prospector[with_pyroma]', 'yapf', 'isort'],
    }
)
