import bpy
from math import pi
from os.path import join, dirname
from pathlib import Path
from mathutils import Euler, Vector, Matrix
import random
import json
import argparse
import sys
import itertools
import functools
import numpy as np
import os
import shutil
import itertools
import math

if __name__ == '__main__':
    # Trigger reimport when the script is run again.
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]

from replicantface.persistent_shuffled_cycle import PersistentShuffledCycle
from replicantface.utils import replicantface_folder


class EnvRandomizer:
    def __init__(self, p_custom_light : float = 0.5):
        # from https://polyhaven.com/hdris/skies/overcast
        # and https://ambientcg.com
        self.p_custom_light = p_custom_light
        self.ENVS = list((replicantface_folder() / 'hdris').glob('*.exr'))
        self.strength_corrections = {
            'EveningEnvironmentHDRI002_1K-HDR.exr': 5.,
            'EveningEnvironmentHDRI004_1K-HDR.exr': 4.,
            'DayEnvironmentHDRI014_1K-HDR.exr' : 4.,
            'EveningSkyHDRI007A_1K-HDR.exr': 3.,
            'IndoorEnvironmentHDRI001_1K-HDR.exr': 12.,
            'IndoorEnvironmentHDRI004_1K-HDR.exr': 5.,
            'IndoorEnvironmentHDRI012_1K-HDR.exr': 2.,
            'IndoorEnvironmentHDRI008_1K-HDR.exr' : 3.,
            'IndoorEnvironmentHDRI005_1K-HDR.exr' : 2.,
            'IndoorEnvironmentHDRI007_1K-HDR.exr' : 3.,
            'IndoorEnvironmentHDRI009_1K-HDR.exr' : 2.,
            'EveningSkyHDRI011A_1K-HDR.exr' : 3.,
            'MorningSkyHDRI007A_1K-HDR.exr': 2.,
            'NightEnvironmentHDRI008_1K-HDR.exr': 8.,
            'NightSkyHDRI003_1K-HDR.exr': 3.,
            'NightSkyHDRI008_1K-HDR.exr': 8.,
            'NightSkyHDRI016B_1K-HDR.exr': 4.,
            'studio_small_03_1k.exr': 0.5,
            'studio_small_04_1k.exr': 0.5,
            'studio_small_06_1k.exr': 0.5,
            'studio_small_07_1k.exr': 0.5,
            'studio_small_09_1k.exr': 0.5,
            'under_bridge_1k.exr' : 0.5,
            'furstenstein_1k.exr' : 10,
            'teutonic_castle_moat_1k.exr' : 0.3,
            'bloem_olive_house_1k.exr' : 0.3,
            'piazza_martin_lutero_1k.exr' : 0.3,
            'docklands_02_1k.exr' : 0.3,
            'little_paris_under_tower_1k.exr' : 0.3,
            'pretville_street_1k.exr' : 0.3,
            'studio_garden_1k.exr' : 0.3,
            'illovo_beach_balcony_1k.exr' : 0.3,
            'preller_drive_1k.exr' : 3.,
            'moon_lab_1k.exr' : 0.5,
            'peppermint_powerplant_1k.exr' : 2.,
            'stierberg_sunrise_1k.exr' : 0.8,
            'snowy_park_01_1k.exr' : 0.7,
            'solitude_night_1k.exr' : 2.
        }
        env_coll = bpy.context.scene.collection.children['EnvCollection']
        assert env_coll.hide_viewport == False
        assert env_coll.hide_render == False
        self._lights = [ o for o in env_coll.objects if o.name.startswith('Light') ]
        self.rotation_helper = env_coll.objects['RotationHelper']

        assert len(self.ENVS) > 10
        self._cycle = PersistentShuffledCycle(self.ENVS, replicantface_folder() / 'state' / 'env.txt')

    def _disable_lights(self):
        for o in self._lights:
            o.hide_set(True)
            o.hide_render = True

    def _randomize_lights(self):
        self._disable_lights()
        assert len(self._lights)==4
        n = np.random.choice([1,2,3], p=[0.8,0.15,0.05])
        chosen = np.random.choice(self._lights, size=n, replace=False)
        for o in chosen:
            o.hide_set(False)
            o.hide_render = False
            o.data.energy = 100 / n
        self.rotation_helper.delta_rotation_euler[2] = random.uniform(-20.,20.)/180.*math.pi

    def _randomize_background(self, strength_scale : float = 1.):
        bg = self._cycle.next()
        self._cycle.save()
        strength = self.strength_corrections.get(bg.name, 1.) * strength_scale

        tex = bpy.context.scene.world.node_tree.nodes["Environment Texture"]
        tex.image = bpy.data.images.load(str(bg), check_existing=True) 
        mat = bpy.context.scene.world.node_tree.nodes["Background"]
        mat.inputs['Strength'].default_value = strength
        texture_coord_mapping = bpy.data.worlds["World"].node_tree.nodes["Mapping"]
        texture_coord_mapping.inputs[2].default_value[2] = random.uniform(0.,math.pi*2.)
    
    def randomize(self):
        if random.uniform(0., 1.) <  self.p_custom_light:
            self._randomize_background(strength_scale=0.05)
            self._randomize_lights()
        else:
            self._randomize_background()
            self._disable_lights()


if __name__ == '__main__':
    scene = EnvRandomizer(p_custom_light=0.5)
    scene.randomize()