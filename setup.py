#!/usr/bin/env python
import os
import sys
from shutil import rmtree

from setuptools import setup, Command

NAME = 'specula'
DESCRIPTION = 'PYramid Simulator Software for Adaptive OpTics Arcetri'
URL = 'https://github.com/FabioRossiArcetri/SPECULA'
EMAIL = 'fabio.rossi@inaf.it'
AUTHOR = 'Fabio Rossi, Alfio Puglisi, Guido Agapito, Lorenzo Busoni, INAF Arcetri Adaptive Optics group'
LICENSE = 'MIT'
KEYWORDS = 'Adaptive Optics, Astrophysics, INAF, Arcetri',

here = os.path.abspath(os.path.dirname(__file__))
# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(name=NAME,
      description=DESCRIPTION,
      version=about['__version__'],
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url=URL,
      author_email=EMAIL,
      author=AUTHOR,
      license=LICENSE,
      keywords=KEYWORDS,
      packages=['specula',
                'specula.data_objects',
                'specula.processing_objects',
                ],
      package_data={
      },
      entry_points={
          'console_scripts': [
              'specula_frontend_start=specula.scripts.web_frontend:start',
              'specula_frontend_stop=specula.scripts.web_frontend:stop',
              'specula=specula.scripts.specula:main',
          ],
      },
      python_requires='>=3.8.0',
      install_requires=["numpy",
                        "scipy",
                        "astropy",
                        "matplotlib",
                        "astro-seeing>=1.1",
                        "symao>=1.0.1",
                        "flask-socketio",
                        "python-socketio",
                        "requests"
                        ],
      extras_require={
          'control': ["iircontrol"]
      },
      include_package_data=True,
      test_suite='test',
      cmdclass={'upload': UploadCommand, },
      )
