from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

version = '0.2.4'

setup(
    name='bindsnet',
    version=version,
    description='Spiking neural networks for ML in Python',
    license='AGPL-3.0',
    long_description=long_description,
    long_description_content_type='text/markdown',  # This is important!
    url='http://github.com/Hananel-Hazan/bindsnet',
    author='Daniel Saunders, Hananel Hazan, Darpan Sanghavi, Hassaan Khan',
    author_email='danjsaund@gmail.com',
    packages=find_packages(),
    zip_safe=False,
    download_url='https://github.com/Hananel-Hazan/bindsnet/archive/%s.tar.gz' % version,
    install_requires=[
        'numpy>=1.14.2', 'torch>=1.0.0', 'tqdm>=4.19.9', 'matplotlib>=2.1.0', 'gym>=0.10.4', 'scikit_image>=0.13.1',
        'scikit_learn>=0.19.1', 'opencv-python>=3.4.0.12', 'pytest>=3.4.0', 'scipy>=1.1.0',
        'cython>=0.28.5', 'pandas>=0.23.4'
    ]
)
