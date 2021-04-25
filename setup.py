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
        "numpy>=1.14.2",
        "torch>=1.5.1",
        "torchvision>=0.6.1",
        "tensorboardX>=1.7",
        "tqdm>=4.19.9",
        "matplotlib>=2.1.0",
        "gym>=0.10.4",
        "scikit-build>=0.11.1",
        "scikit_image>=0.13.1",
        "scikit_learn>=0.19.1",
        "opencv-python>=3.4.0.12",
        "pytest>=3.4.0",
        "scipy>=1.1.0",
        "cython>=0.28.5",
        "pandas>=0.23.4",
    ],
)
