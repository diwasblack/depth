from setuptools import setup, find_packages

setup(
    name='depth',
    author='Diwas Sharma',
    author_email='diwasblack@gmail.com',
    version='0.0.dev0',
    packages=find_packages(),
    license='MIT',
    install_requires=['numpy>=1.14.5'],
    test_suite="tests"
)
