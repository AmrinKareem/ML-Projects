from setuptools import find_packages, setup
from typing import List
HYPHENE_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    """This function returns a list of requirements"""
    reqs = []
    with open(file_path) as f:
        reqs=f.readlines()
        reqs=[req.replace("\n", "") for req in reqs]
        if HYPHENE_DOT in reqs:
            reqs.remove(HYPHENE_DOT)
    return reqs
    
setup(
    name="mlproject",
    version='0.0.1',
    author="amrin",
    author_email="amrinkareem.mec@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)