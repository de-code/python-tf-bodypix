from time import time

from setuptools import find_packages, setup


with open('requirements.txt', 'r') as f:
    REQUIRED_PACKAGES = f.readlines()


with open('README.md', 'r') as f:
    long_description = f.read()


def local_scheme(*_, **__):
    return str(int(time()))


packages = find_packages()
setup(
    name="tf-bodypix",
    use_scm_version={
        "local_scheme": local_scheme
    },
    setup_requires=['setuptools_scm'],
    author="Daniel Ecer",
    url="https://github.com/de-code/python-tf-bodypix",
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    include_package_data=True,
    description='Python implemention of the TensorFlow BodyPix model.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
