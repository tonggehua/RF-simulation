from typing import Tuple, Union
import setuptools


setuptools.setup(
    author="Gehua Tong",
    author_email="gt2407@columbia.edu",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    description="MRI RF Simulation",
    include_package_data=True,
    install_requires=[
        'matplotlib>=3.4.2',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pypulseq==1.3.1.post1'
    ],
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    name="RF-simulation",
    packages=setuptools.find_packages(),
    python_requires='>=3.7.0',
    url="https://github.com/tonggehua/RF-simulation",
    version="1.0.0",
)
