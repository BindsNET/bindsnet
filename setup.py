from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="BindsNET",
    version="0.2.9",
    description="Spiking neural networks for ML in Python",
    license="AGPL-3.0",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="http://github.com/Hananel-Hazan/bindsnet",
    author="Hananel Hazan, Daniel Saunders, Darpan Sanghavi, Hassaan Khan",
    author_email="hananel@hazan.org.il",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "scipy>=1.14.1,"
        "numpy>=2.1.0,"
        "cython>=3.0.11,"
        "torch==2.4.0,"
        "torchvision==0.19.0,"
        "tensorboardX==2.6.2.2,"
        "tqdm>=4.66.5,"
        "setuptools>=74.1.1,"
        "matplotlib>=3.9.2,"
        "gym>=0.26.2,"
        "scikit-build>=0.18.0,"
        "scikit_image>=0.24.0,"
        "scikit_learn>=1.5.1,"
        "opencv-python>=4.10.0.84,"
        "pytest>=8.3.2,"
        "pandas>=2.2.2,"
    ],
)
