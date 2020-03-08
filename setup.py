from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="BindsNET",
    version = "0.2.7",
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
        "foolbox",
    ],
)
