from setuptools import setup
import setuptools

setup(
    name='src',
    version='1.0',
    packages=['tests', 'src'],
    install_requires=['loguru==0.5.3', 'mongoengine==0.23.1',
                      'mongomock==3.23.0', 'numpy==1.21.1', 'Pillow==8.3.1',
                      'pymongo==3.12.0', 'pytest==6.2.4', 'toml==0.10.2', 'torch',
                      'torchvision',
                      'trains==0.16.4', 'pytorch-lightning', 'simpleitk', 'lightning-bolts'],
    url='',
    license='',
    author='ohad',
    author_email='ogdoron@gmail.com',
    description='Main Project'
)
