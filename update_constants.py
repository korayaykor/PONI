"""
Script to integrate HM3D dataset constants into the main constants.py file.
This can be run once to update the constants.py file.
"""

import os
import re

def update_constants_file():
    """Update the constants.py file to include HM3D dataset."""
    # Read the current constants.py file
    with open('poni/constants.py', 'r') as f:
        constants_content = f.read()
    
    # Read the HM3D constants file
    with open('poni/constants_hm3d.py', 'r') as f:
        hm3d_content = f.read()
    
    # Extract key parts from HM3D constants
    hm3d_registered_datasets = re.search(r'REGISTERED_DATASETS = \[(.*?)\]', hm3d_content, re.DOTALL).group(1)
    hm3d_split_scenes = re.search(r'SPLIT_SCENES = \{(.*?)"hm3d": \{(.*?)\}', hm3d_content, re.DOTALL).group(2)
    hm3d_object_categories = re.search(r'OBJECT_CATEGORIES = \{(.*?)"hm3d": \[(.*?)\],', hm3d_content, re.DOTALL).group(2)
    
    # Update REGISTERED_DATASETS
    constants_content = re.sub(
        r'REGISTERED_DATASETS = \[(.*?)\]',
        f'REGISTERED_DATASETS = [{hm3d_registered_datasets}]',
        constants_content,
        flags=re.DOTALL
    )
    
    # Update SPLIT_SCENES
    constants_content = re.sub(
        r'SPLIT_SCENES = \{(.*?)\}',
        f'SPLIT_SCENES = {{\g<1>,"hm3d": {{   {hm3d_split_scenes}    }}}}',
        constants_content, 
        flags=re.DOTALL
    )
    
    # Update OBJECT_CATEGORIES
    constants_content = re.sub(
        r'OBJECT_CATEGORIES = \{(.*?)\}',
        f'OBJECT_CATEGORIES = {{\g<1>,"hm3d": [{hm3d_object_categories}]}}',
        constants_content,
        flags=re.DOTALL
    )
    
    # Write the updated constants file
    with open('poni/constants_updated.py', 'w') as f:
        f.write(constants_content)
    
    print("Updated constants file written to 'poni/constants_updated.py'")
    print("Review the changes and rename to 'constants.py' if satisfied.")

if __name__ == "__main__":
    update_constants_file()