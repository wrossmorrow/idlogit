from setuptools import setup

setup(name='idlogit',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/wrossmorrow/idlogit',
      author='W. Ross Morrow',
      author_email='morrowwr@gmail.com',
      license='MIT',
      packages=['idlogit'],
      install_requires=['numpy','scipy','ecos'],
      zip_safe=False)