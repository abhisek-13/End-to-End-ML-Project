from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath:str)->List[str]:
  # This function will return the list of requirements.
  hypen_e_dot = '-e .'
  requirements = []
  with open(filepath) as file_obj:
    requirements = file_obj.readlines()
    requirements = [req.replace('\n','') for req in requirements]
    if hypen_e_dot in requirements:
      requirements.remove(hypen_e_dot)
    
  return requirements

setup(
  name = 'end to end ml project',
  version= '0,0,1',
  author='abhisek',
  author_email='abhisekmaharana9861@gmail.com',
  packages=find_packages(),
  install_requires = get_requirements('requirements.txt')
)