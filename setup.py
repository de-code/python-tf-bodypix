from time import time

from setuptools import find_packages, setup


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


def get_requirement_groups(requirement):
    if 'tensorflow' in requirement:
        return ['tf']
    if 'tfjs' in requirement:
        return ['tfjs']
    return [None]


def get_required_and_extras(all_required_packages):
    grouped_extras = {}
    for requirement in all_required_packages:
        for group in get_requirement_groups(requirement):
            grouped_extras.setdefault(group, []).append(requirement)
    return (
        grouped_extras[None],
        {key: value for key, value in grouped_extras.items() if key}
    )


DEFAULT_REQUIRED_PACKAGES, EXTRAS = get_required_and_extras(REQUIRED_PACKAGES)


packages = find_packages()
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
