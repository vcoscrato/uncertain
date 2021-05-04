from setuptools import find_packages, setup

setup(
    name='uncertain',
    version='0.7.4',
    packages=find_packages(),
    install_requires=['torch>=1.7.0', 'numpy'],
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)