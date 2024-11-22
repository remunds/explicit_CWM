import pathlib
import setuptools
from setuptools import find_namespace_packages

#both requirements.tx
dreamer_req = pathlib.Path('dreamerv3/requirements.txt').read_text().splitlines()
embodied_req = pathlib.Path('embodied/requirements.txt').read_text().splitlines()
req = dreamer_req + embodied_req

setuptools.setup(
    name='dreamerv3',
    version='1.5.0',
    description='Mastering Diverse Domains through World Models',
    author='Danijar Hafner',
    url='http://github.com/danijar/dreamerv3',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(exclude=['example.py']),
    include_package_data=True,
    install_requires=req,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
