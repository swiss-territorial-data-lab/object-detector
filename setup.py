from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='stdl-object-detector',
    version='latest',
    description='A suite of Python scripts allowing the end-user to use Deep Learning to detect objects in georeferenced raster images.',
    author='Swiss Territorial Data Lab (STDL)',
    author_email='info@stdl.ch',
    python_requires=">=3.8",
    license="MIT license",
    entry_points = {
        'console_scripts': [
            'generate_tilesets=scripts.generate_tilesets:main',
            'train_model=scripts.train_model:main',
            'make_predictions=scripts.make_predictions:main',
            'assess_predictions=scripts.assess_predictions:main',
            ]
    },
    install_requires=requirements
)