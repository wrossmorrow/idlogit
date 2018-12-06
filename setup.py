from setuptools import setup

setup(name='idlogit',
      version='0.1',
      description='estimate Logit models with "idiosyncratic deviations"',
      url='https://github.com/wrossmorrow/idlogit',
      author='W. Ross Morrow',
      author_email='morrowwr@gmail.com',
      license='Apache2',
      packages=['idlogit'],
      install_requires=['numpy','scipy','ecos'],
      zip_safe=False)