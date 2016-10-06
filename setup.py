# setup code taken from https://www.jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/

from setuptools import setup
from setuptools.command.test import test as TestCommand
import io
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

# long_description = read('README.rst', 'CHANGES.txt')
long_description = read('README.rst')


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='ReinforcePy',
    version=0.2,
    url='http://github.com/islandman93/reinforcepy',
    license='GNU License',
    author='IslandMan93',
    tests_require=['pytest'],
    install_requires=['numpy>=1.8.x', 'tensorflow>=0.9', 'tflearn>=0.2', 'sacred>=0.6.10'],
    cmdclass={'test': PyTest},
    author_email='islandman93@gmail.com',
    description='Collection of reinforcement learners implemented in python.',
    long_description=long_description,
    packages=['reinforcepy'],
    include_package_data=True,
    platforms='any',
    test_suite='reinforcepy.test.test_reinforcepy',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: Alpha',
        'Natural Language :: English',
        'Environment :: Machine Learning',
        'Intended Audience :: Developers/Researchers',
        'License :: GNU License',
        'Operating System :: OS Independent',
        # 'Topic :: Software Development :: Libraries :: Python Modules',
        # 'Topic :: Software Development :: Libraries :: Application Frameworks',
        # 'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        ],
    extras_require={
        'testing': ['pytest'],
    }
)
