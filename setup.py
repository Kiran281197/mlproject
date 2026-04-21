from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    """
        This function will return list of requirements.
    """
    with open("requirements.txt", "r") as file:
        libraries = file.readlines()
        libraries = [lib.replace("\n","")for lib in libraries]

        if HYPHEN_E_DOT in libraries:
            libraries.remove(HYPHEN_E_DOT)

    return libraries

setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Kiran",
    author_email = "kirandata97@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements("requirements.txt")
)

