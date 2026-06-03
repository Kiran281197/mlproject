from setuptools import find_packages, setup
from typing import List

"""
    setup.py file is used for packaging the entire application, distribute and store metadata related to application
"""

def get_requirements(file_path:str)->List[str]:
    '''
        This fucntion will return the list of libraries from give file path
    '''
    requirements = []
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements if req!="-e ."]
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="kiran",
    author_email="1997kiranreddy@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)



