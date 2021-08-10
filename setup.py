import sys
from setuptools import setup, find_packages

if sys.version_info >= (3, 10, 0) or sys.version_info < (3, 7, 0):
    raise OSError(f'PQlite requires Python 3.7/3.8/3.9, but yours is {sys.version}')

try:
    with open('README.md', encoding='utf8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

extras = {}
extras['testing'] = ['pytest']

setup(
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
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
    package_dir={'': 'pqlite'},
    packages=find_packages(
        exclude=['*.tests', '*.tests.*', 'tests.*', 'tests', 'test', 'docs', 'src']
    ),
    zip_safe=False,
    keywords='pqlite product-quantization approximate-nearest-neighbor ivf-pq',
)
