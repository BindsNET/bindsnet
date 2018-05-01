from setuptools import setup, find_packages

setup(name='bindsnet',
      version='0.1',
      description='Spiking neural networks for ML in Python',
      url='http://github.com/Hananel-Hazan/bindsnet',
      author='Daniel Saunders, Hananel Hazan, Darpan Sanghavi, Hassaan Khan',
      author_email='djsaunde@cs.umass.edu',
      license='MIT',
      packages=['bindsnet'],
	  zip_safe=False,
	  install_requires=['numpy==1.14.2',
						'torch>=0.4.0',
						'tqdm>=4.19.9',
						'matplotlib>=2.1.0',
						'gym>=0.10.4',
						'scikit_image>=0.13.1',
						'scikit_learn>=0.19.1',
                     'opencv-python==3.4.0.12'])
