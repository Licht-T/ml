from setuptools import setup, find_packages

setup(
    name='ml',
    version='1.0.0',
    packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [],
    },
    test_suite='pytest',
    tests_require=[
        'pytest',
        'pytest-cov'
    ]
)
