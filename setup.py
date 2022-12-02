from setuptools import setup, find_packages

setup(
    name = 'Smart Process Analytics',
    author = 'Original by Vicky Sun, modified by Pedro Seber',
    author_email = 'pseber@mit.edu',
    license = 'MIT',
    version = '1.1.0',
    description = 'A software for predictive modeling.',
    zip_safe = False,
    packages = find_packages(where = 'Code-SPA', include = ['SPA.py']),
    package_dir = {'': 'Code-SPA'},
    # Note: other versions of these packages most likely still work, but we have not determined what are the minimum version requirements
    install_requires = [
        'notebook==6.4.11',
        'scikit-learn==1.0.2',
        'matplotlib==3.5.1',
        'pandas==1.4.2',
        'statsmodels==0.13.2',
        'seaborn==0.11.2',
        'rpy2==3.5.1',
        'openpyxl==3.0.9',
        'xlrd==2.0.1',
        'tensorflow==2.8.1'
        ],
    )

