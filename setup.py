from time import time

from setuptools import find_packages, setup

from tf_bodypix.utils.dist import (
    get_requirements_with_groups,
    get_required_and_extras
)


with open('requirements.txt', 'r') as f:
    REQUIRED_PACKAGES = f.readlines()


with open('README.md', 'r') as f:
    LONG_DESCRIPTION = '\n'.join([
        line.rstrip()
        for line in f
        if not line.startswith('[![')
    ])


def local_scheme(version):
    if not version.distance and not version.dirty:
        return ""
    return str(int(time()))


DEFAULT_REQUIRED_PACKAGES, EXTRAS = get_required_and_extras(
    get_requirements_with_groups(REQUIRED_PACKAGES)
)


packages = find_packages(exclude=["tests", "tests.*"])

setup(
    name="tf-bodypix",
    use_scm_version={
        "local_scheme": local_scheme
    },
    setup_requires=['setuptools_scm'],
    author="Daniel Ecer",
    url="https://github.com/de-code/python-tf-bodypix",
    install_requires=DEFAULT_REQUIRED_PACKAGES,
    extras_require=EXTRAS,
    packages=packages,
    include_package_data=True,
    description='Python implemention of the TensorFlow BodyPix model.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
