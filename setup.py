from setuptools import setup, find_packages

setup(
    name = 'Smart Process Analytics',
    author = 'Pedro Seber, forked from code by Vicky Sun',
    author_email = 'pseber@mit.edu',
    license = 'MIT',
    version = '1.4.2',
    description = 'Smart Process Analytics (SPA) is a software package for automatic machine learning. Given user-input data (and optional user preferences), SPA automatically cross-validates and tests ML and DL models. Model types are selected based on the properties of the data, minimizing the risk of data-specific variance.',
    zip_safe = False,
    packages = find_packages(where = 'Code-SPA', include = ['SPA.py']),
    package_dir = {'': 'Code-SPA'},
    # Note: other versions of these packages most likely also work, but we have not determined what are the minimum version requirements
    install_requires = [
        'notebook>=6.4.11',
        'scikit-learn>=1.0.2',
        'matplotlib>=3.5.1',
        'pandas>=1.4.2',
        'statsmodels>=0.13.2',
        'seaborn>=0.11.2',
        'openpyxl>=3.0.9',
        'xlrd>=2.0.1',
        'ace-cream>=0.4', # Comment out this line if you are on Windows and having trouble installing ace-cream.
        'torch>=2.0.0'
        ],
    )
