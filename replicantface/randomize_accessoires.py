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


from replicantface.utils import hide_object, HeadCoverage, update_child_of_constraint, find_hum


class Accessoires:
    def __init__(self, p_glasses = 1., p_hat = 1., p_phones = 1., p_mask = 1., root : bpy.types.Object | None = None):
        self.root = root if root is not None else bpy.context.scene.objects['accessoires']
        self.accessoires = { o.name:o for o in self.root.children }
        self.p_glasses = p_glasses
        self.p_hat = p_hat
        self.p_phones = p_phones
        self.p_mask = p_mask
        self._all_glasses = [ k for k in self.accessoires.keys() if k.startswith('glasses') ]
        self._all_hats = [ k for k in self.accessoires.keys() if (k.startswith('hat') or k.startswith('special')) ]
        self._all_phones = [ k for k in self.accessoires.keys() if k.startswith('headphones') ]
        self._all_masks = [ k for k in self.accessoires.keys() if k.startswith('mask') ]


    def _accept_selection(self, phones : str, hat : str, glasses: str, mask : str):
        if phones and hat not in ['', 'hat36', 'hat18', 'hat17', 'hat39', 'special5']:
            return False
        if glasses and hat in [ 'hat7', 'hat19', 'hat27', 'hat35', 'special1', 'special2', 'hat40' ]:
            return False
        if mask == 'mask2' and hat.startswith('special'):
            return False
        if mask == 'mask2' and hat in [ 'hat27', 'hat22' , 'hat19', 'hat16', 'hat15', 'hat14', 'hat5' ,'hat4']:
            return False
        if mask == 'mask1' and hat in [ 'hat7', 'special1', 'special2', 'hat16']:
            return False

        # TODO disable mask with strong deformed faces?
        # TODO texture pattern on some helmets and masks
        return True


    def _sample_accessoir_selection(self) -> tuple[str,...]:
        for _ in range(10):
            glasses_selected = ''
            if self.p_glasses > 0.  and random.uniform(0.,1.) < self.p_glasses:
                glasses_selected = random.choice(self._all_glasses)
            hat_selected = ''
            if self.p_hat > 0. and random.uniform(0.,1.) < self.p_hat:
                hat_selected : str = random.choice(self._all_hats)
            phones_selected = ''
            if self.p_phones > 0. and random.uniform(0.,1.) < self.p_phones:
                phones_selected : str = random.choice(self._all_phones)
            mask_selected = ''
            if self.p_mask > 0. and random.uniform(0.,1.) < self.p_mask:
                mask_selected : str = random.choice(self._all_masks)
            # Break once valid configuration is found
            if self._accept_selection(phones_selected, hat_selected, glasses_selected, mask_selected):
                break
        objs = (glasses_selected, hat_selected, phones_selected, mask_selected)
        return tuple(filter(lambda x: x, objs))


    def _get_bsdf(self, material_name : str) -> bpy.types.ShaderNodeBsdfPrincipled:
        mat = bpy.data.materials[material_name]
        return mat.node_tree.nodes['Principled BSDF']


    def enable_randomly(self):
        for v in self.accessoires.values():
            hide_object(v, True)
        
        object_names = self._sample_accessoir_selection()
        objects=[self.accessoires[name] for name in object_names]

        head_covered = any(o.startswith('hat') for o in object_names) and not any(o in ('hat39', 'hat24') for o in object_names)
        head_covered = head_covered or any(o.startswith('special') for o in object_names)

        hair_banned = {
            'hat40', 'hat36', 'special3', 'special2', 'special1', 'hat26', 'hat19', 'hat16', 'hat12', 'hat11', 'hat9', 'hat8', 'hat7', 
        }

        beard_banned = {
            'special3', 'special2', 'special1', 'mask2', 'mask1', 'hat16', 
        }

        hair_allowed=not any((o in hair_banned) for o in object_names)
        allow_beard=not any((o in beard_banned) for o in object_names)

        for obj in objects:
            hide_object(obj, False)

        return HeadCoverage(
            top_covered='tight' if not hair_allowed else ('loose' if head_covered else 'none'),
            allow_beard=allow_beard
        )


    def randomize_materials(self):
        # Glass
        glasses_shade = random.betavariate(0.2, 1.)
        glasses_metallic = 0.8 if random.randint(0,5)==0 else 0.
        mat_glass = bpy.data.materials['accessoire_glass']
        bsdf : bpy.types.ShaderNodeBsdfPrincipled = mat_glass.node_tree.nodes['Principled BSDF']
        bsdf.inputs['Coat Weight'].default_value = glasses_shade
        bsdf.inputs['Metallic'].default_value = glasses_metallic
        # Other materials. Randomize with little regard to realism ...
        for name in [
            'accessoire_white',
            'accessoire_grey',
            'accessoire_black',
            'accessoire_glasses_frame',
            'accessoire_hat_plastic_or_metal']:
            bsdf = self._get_bsdf(name)
            metallic = random.uniform(0., 1.)
            roughness = random.uniform(0.,1. if metallic < 0.5 else 0.5)
            bsdf.inputs['Base Color'].default_value = (*colorsys.hsv_to_rgb(
                h=random.uniform(0.,1.),
                s=random.uniform(0.,1.),
                v=random.uniform(0.,1.),
            ), 1.)
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
        for name in ['accessoire_black', 'accessoire_hat_black']:
            bsdf = self._get_bsdf(name)
            bsdf.inputs['Base Color'].default_value = (*colorsys.hsv_to_rgb(
                h=0.,
                s=0.,
                v=random.uniform(0.,0.2),
            ), 1.)
            bsdf.inputs['Metallic'].default_value = 0.
            bsdf.inputs['Roughness'].default_value = random.uniform(0.5,1.)
        for name in ['accessoire_hat_white', 'accessoire_hat_grey', 'accessoire_hat_other']:
            bsdf = self._get_bsdf(name)
            bsdf.inputs['Base Color'].default_value = (*colorsys.hsv_to_rgb(
                h=random.uniform(0.,1.),
                s=random.uniform(0.,1.),
                v=random.uniform(0.,1.),
            ), 1.)
            bsdf.inputs['Metallic'].default_value = 0.
            bsdf.inputs['Roughness'].default_value = random.uniform(0.5,1.)


if __name__ == '__main__':
    hum = find_hum()
    accessoires = Accessoires(0.8,0.8,0.8, 0.1)
    update_child_of_constraint(accessoires.root, hum.objects.rig, 'head')
    hc = accessoires.enable_randomly()
    print (hc)
    accessoires.randomize_materials()
    bpy.context.view_layer.update()
