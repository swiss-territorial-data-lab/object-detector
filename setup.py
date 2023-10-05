from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='stdl-object-detector',
    version='1.0.0',
    description='A suite of Python scripts allowing the end-user to use Deep Learning to detect objects in georeferenced raster images.',
    author='Swiss Territorial Data Lab (STDL)',
    author_email='info@stdl.ch',
    python_requires=">=3.8",
    license="MIT license",
    entry_points = {
        'console_scripts': [
            'stdl-objdet=scripts.cli:main'
            ]
    },
    install_requires=requirements,
    packages=find_packages()
)