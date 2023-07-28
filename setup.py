from setuptools import setup

setup(
    name='cryptobazen_statistics',
    version='0.0.2',
    description='Statistic functions for Transformer models',
    long_description='The PyToch Transformer model needs sometimes a custom statistic function. This package contains some of these functions.',
    author='Cryptobazen',
    author_email='erwinvink@outlook.com',
    packages=["cryptobazen_statistics"],
    license='MIT License',
    install_requires=[
        'numpy>=1.24',
        'scikit-learn>=1.2.0'
    ],
    zip_safe=False,
)
