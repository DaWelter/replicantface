import bpy
from math import pi
from os.path import join, dirname
from pathlib import Path
from mathutils import Euler, Vector, Matrix
from HumGen3D import Human
from HumGen3D.common import is_part_of_human
import random
import json
import argparse
import sys
import itertools
import functools
import numpy as np
import os
import shutil
from typing import NamedTuple
import colorsys


if __name__ == '__main__':
    # Trigger reimport when the script is run again.
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]


from replicantface.utils import replicantface_folder


def add_accessories(filename):
    """Merges objects from the scene given by filename into the Accessoires group."

    For loading additional assets into the main scene. The asset scene should contain all assets
    at the scene root level and only the assets there.
    """
    collection = bpy.data.collections["Accessoires"]
    parent = collection.objects["accessoires"]
    existing_objects = { o.name for o in collection.objects }

    with bpy.data.libraries.load(str(filename)) as (data_from, data_to):
        [ data_to.objects.append(name) for name in data_from.objects if name not in existing_objects ]
    
    for obj in data_to.objects:
        collection.objects.link(obj)
        obj.parent = parent

    # Remove duplicated materials.
    # This is needed because randomization uses hardcoded names, and loading the same names again
    # will cause duplicated materials to be created. They have numeric suffixes so they are easy to find.
    mat_names = { x.material.name for obj in data_to.objects for x in obj.material_slots }
    mats = bpy.data.materials
    for mat in mats:
        (original, _, ext) = mat.name.rpartition(".")
        if mat.name in mat_names and ext.isnumeric() and mats.find(original) != -1:
            #print("%s -> %s" %(mat.name, original))
            mat.user_remap(mats[original])
            mats.remove(mat)


if __name__ == '__main__':
    filepath = replicantface_folder() / "assets" / "all-accessories.blend"
    add_accessories(filepath)