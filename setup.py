import os
import sys
from distutils.sysconfig import get_python_inc

import numpy as np
import pybind11
import setuptools
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

if sys.version_info >= (3, 10, 0) or sys.version_info < (3, 7, 0):
    raise OSError(f'PQlite requires Python 3.7/3.8/3.9, but yours is {sys.version}')

include_dirs = [pybind11.get_include(), np.get_include()]

libraries = []
extra_objects = []

try:
    pkg_name = 'pqlite'
    libinfo_py = os.path.join(pkg_name, '__init__.py')
    libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
    version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][
        0
    ]
    exec(version_line)  # produce __version__
except FileNotFoundError:
    __version__ = '0.0.0'

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

try:
    with open('requirements.txt') as f:
        base_deps = f.read().splitlines()

    # remove blank lines and comments
    base_deps = [
        x.strip()
        for x in base_deps
        if ((x.strip()[0] != '#') and (len(x.strip()) > 3) and '-e git://' not in x)
    ]
except FileNotFoundError:
    base_deps = []

COMPILER_DIRECTIVES = {
    'language_level': -3,
    'embedsignature': True,
    'annotation_typing': False,
}

ext_modules = [
    Extension(
        'pqlite.hnsw_bind',
        ['./bindings/hnsw_bindings.cpp'],
        include_dirs=include_dirs + ['./include/hnswlib'],
        libraries=libraries,
        language='c++',
        extra_objects=extra_objects,
    )
] + cythonize(
    [
        Extension(
            'pqlite.pq_bind',
            ['./bindings/pq_bindings.pyx'],
            include_dirs=include_dirs + [get_python_inc(plat_specific=True)],
            libraries=libraries,
            language='c++',
            extra_objects=extra_objects,
        ),
    ],
    compiler_directives=COMPILER_DIRECTIVES,
)
#
# ext_modules = cythonize([
#                 Extension(
#                     'pqlite.pq_bind',
#                     ['./bindings/pq_bindings.pyx'],
#                     include_dirs=include_dirs + [get_python_inc(plat_specific=True)],
#                     libraries=libraries,
#                     language='c++',
#                     extra_objects=extra_objects,
#                 ),
#             ], compiler_directives=COMPILER_DIRECTIVES)

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError(
            'Unsupported compiler -- at least C++11 support ' 'is needed!'
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        'msvc': ['/EHsc', '/openmp', '/O2'],
        'unix': ['-O3', '-march=native'],  # , '-w'
    }
    link_opts = {
        'unix': [],
        'msvc': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        link_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    else:
        c_opts['unix'].append('-fopenmp')
        link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            # if has_flag(self.compiler, '-fvisibility=hidden'):
            #     opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))

        build_ext.build_extensions(self)


extras = {}
extras['testing'] = ['pytest']

# for e in ext_modules:
#     e.cython_directives = COMPILER_DIRECTIVES

setup(
    name='pqlite',
    version=__version__,
    description='Fast and Light Approximate Nearest Neighbor Search Database integrated with the Jina Ecosystem',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='Jina AI',
    author_email='team@jina.ai',
    url='https://github.com/jinaai/pqlite',
    download_url='https://github.com/jinaai/pqlite/tags',
    license='Apache License 2.0',
    extras_require=extras,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    package_data={'bindings': ['*.pyx', '*.pxd', '*.pxi']},
    install_requires=base_deps,
    setup_requires=['setuptools>=18.0', 'wheel', 'cython'],
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
    packages=find_packages(
        exclude=[
            '*.tests',
            '*.tests.*',
            'tests.*',
            'tests',
            'test',
            'docs',
            'src',
            'executor',
        ]
    ),
    zip_safe=False,
    keywords='product-quantization approximate-nearest-neighbor',
)
