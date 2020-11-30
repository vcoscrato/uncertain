from setuptools import find_packages, setup

setup(
    name='uncertain',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['torch>=1.7.0'],
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)