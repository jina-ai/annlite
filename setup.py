import sys
from setuptools import setup, find_packages
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
from distutils.sysconfig import get_python_inc


if sys.version_info >= (3, 10, 0) or sys.version_info < (3, 7, 0):
    raise OSError(f'PQlite requires Python 3.7/3.8/3.9, but yours is {sys.version}')

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

extras = {}
extras['testing'] = ['pytest']

####### Cython begin #######
include_dirs = [
        numpy.get_include(),
        get_python_inc(plat_specific=True),
]
MOD_NAMES = ['pqlite.utils.asymmetric_distance']

ext_modules = []
for name in MOD_NAMES:
    mod_path = name.replace(".", "/") + ".pyx"
    ext = Extension( name, [mod_path], language="c++", include_dirs=include_dirs, extra_compile_args=["-std=c++11"])
    ext_modules.append(ext)

COMPILER_DIRECTIVES = {
    "language_level": -3,
    "embedsignature": True,
    "annotation_typing": False,
}
####### Cython end #######

setup(
    ext_modules=cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES),
    include_dirs=include_dirs,
    name='pqlite',
    version='0.0.1',
    description='Blaze Fast and Light Approximate Nearest Neighbor Search Database',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='Jina AI',
    author_email='team@jina.ai',
    url='https://github.com/jinaai/pqlite',
    download_url='https://github.com/jinaai/pqlite/tags',
    license='Apache License 2.0',
    extras_require=extras,
    setup_requires=['setuptools>=18.0', 'wheel'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    # package_dir={'': 'pqlite'},
    packages=find_packages(
        exclude=['*.tests', '*.tests.*', 'tests.*', 'tests', 'test', 'docs', 'src']
    ),
    #package_data={"": ["*.pyx", "*.pxd", "*.pxi"]},
    zip_safe=False,
    keywords='pqlite product-quantization approximate-nearest-neighbor ivf-pq',
)
