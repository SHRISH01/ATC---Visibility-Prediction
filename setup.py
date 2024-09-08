from setuptools import setup, find_packages 
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path ) as f:
        requirements = f.readlines()
        requirements = [req.replace("/n","")for req in requirements]


        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)



    return requirements


setup(name='Visibility_Prediction' ,
      version='1.0',
      author='RA2332014010177_RA2332014010178',
      author_email='shrishkamboz@gmail.com',
      packages= find_packages(),
      install_requires=get_requirements('requirements.txt')
      )