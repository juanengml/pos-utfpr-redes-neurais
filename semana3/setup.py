from setuptools import setup, find_packages

setup(
    name='house_sales_predictor',
    version='0.1.0',
    description='A machine learning project to predict house sales prices using MLPRegressor',
    author='Seu Nome',
    author_email='juanengml@gmail.com',
    url='https://github.com/juanengml/house-sales-predictor',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
