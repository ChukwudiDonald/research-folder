from setuptools import setup, find_packages

setup(
    name ='pyserialview',
    version ='0.1.0',
    description ="A lightweight, flexible framework for acquiring, logging, and live-plotting data from serial devices.",
    long_description = open('README.md').read(),
    long_description_content_type ='text/markdown',
    packages = find_packages(),
    python_requires = '>=3.6',
    install_requires =[
            'pyserial>=3.5',
            'matplotlib>=3.3',
            'pandas>=1.1',
            'pyyaml>=6.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],

)

