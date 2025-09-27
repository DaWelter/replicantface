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
from typing import Sequence, cast

if __name__ == '__main__':
    sys.path.append(Path(bpy.data.filepath).parent.as_posix())
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]

from replicantface.persistent_shuffled_cycle import PersistentShuffledCycle


CLOTHES_FEMALE = [
    # The other presets had bad intersection with the body
    'outfits/female/Winter/Frosty_Evening.blend',
    'outfits/female/Casual/Skinny_Look.blend',
    'outfits/female/Casual/Relaxed_Weekday.blend',
    'outfits/female/Casual/Stylish_Casual.blend',
    'outfits/female/Casual/Smart_Casual.blend',
    'outfits/female/Extra Outfits Pack/Pirate.blend',
    'outfits/female/Extra Outfits Pack/CEO.blend',
    'outfits/female/Extra Outfits Pack/Flight Suit.blend',
    'outfits/female/Extra Outfits Pack/Presentation.blend',
    'outfits/female/Extra Outfits Pack/Lab Tech.blend',
    'outfits/female/Extra Outfits Pack/Dress.blend',
    'outfits/female/Extra Outfits Pack/Kimono.blend',
    'outfits/female/Office/Relaxed_Dresscode.blend',
]

CLOTHES_MALE = [
    'outfits/male/Winter/Frosty_Evening.blend',
    'outfits/male/Casual/Casual_Weekday.blend',
    'outfits/male/Casual/Relaxed_Office.blend',
    'outfits/male/Casual/Skinny_Look.blend',
    'outfits/male/Casual/Stylish_Casual.blend',
    'outfits/male/Casual/Smart_Casual.blend',
    'outfits/male/Extra Outfits Pack/Pirate.blend',
    'outfits/male/Extra Outfits Pack/Flight Suit.blend',
    'outfits/male/Extra Outfits Pack/Lab Tech.blend',
    'outfits/male/Extra Outfits Pack/Relaxed Fit.blend',
    'outfits/male/Extra Outfits Pack/On the road.blend',
    'outfits/male/Extra Outfits Pack/Bomber look.blend',
    'outfits/male/Summer/Beach_Day.blend',
    'outfits/male/Summer/BBQ_Barry.blend',
    'outfits/male/Summer/Office_Excursion.blend', 
    'outfits/male/Office/Open_Suit.blend',
    'outfits/male/Office/New_Intern.blend',
    'outfits/male/Office/Summer_Lawyer.blend',
    'outfits/male/Office/Stock_Exchange.blend',
    'outfits/male/Office/Relaxed_Dresscode.blend',
]

class ClothesRandomizer:
    def __init__(self, p_naked : float, p_pattern : float):
        dir = Path(bpy.data.filepath).parent
        self._preset_cycle = {
            'male' : PersistentShuffledCycle(CLOTHES_MALE, dir / 'state' / 'clothes_male.txt'),
            'female' : PersistentShuffledCycle(CLOTHES_FEMALE, dir / 'state' / 'clothes_female .txt')
        }
        self._p_naked = p_naked
        self._p_pattern = p_pattern

    def randomize(self, hum : Human):
        if random.uniform(0.,1.) < self._p_naked:
            # Caution: outfit removal does not work correctly
            #          But since this script is one shot, it can start with a naked human.
            #hum.clothing.outfit.remove()
            pass
        else:
            preset = self._preset_cycle[hum.gender].next()
            self._preset_cycle[hum.gender].save()

            print ("Applying clothes preset: ", preset)

            hum.clothing.outfit.set(preset, context = bpy.context)

            objects = cast(Sequence[bpy.types.Object], hum.clothing.outfit.objects)
            for o in objects:
                for m in o.data.materials:
                    nodes = m.node_tree.nodes
                    for input in nodes["HG_Control"].inputs:
                        if input.name.lower().startswith('main color'):
                            input.default_value = (*colorsys.hsv_to_rgb(
                                h=random.uniform(0.,1.),
                                s=random.betavariate(0.5,0.5),
                                v=random.betavariate(0.5,0.5),
                            ),1.)
                        if input.name.lower().startswith('normal strength'):
                            input.default_value = random.uniform(1.,5.)
                        if input.name.lower().startswith('wear amount'):
                            input.default_value = random.uniform(0.,1.)

            for o in objects:
                if random.uniform(0.,1.) < self._p_pattern:
                    hum.clothing.outfit.pattern.set_random(o, context=bpy.context)
            
            # Delete unwanted objects. The pirate outfit has a hat. It's the only one with
            # accessoirs which are out of scope for now so the hat is deleted.
            to_remove = [ o for o in bpy.context.scene.objects if o.name.startswith('HG_Pirate_Hat') ]
            with bpy.context.temp_override(selected_objects = to_remove):
                bpy.ops.object.delete()


if __name__ == '__main__':
    hum = Human.from_existing(bpy.context.object)
    randomizer = ClothesRandomizer(p_naked=0., p_pattern=0.1)
    randomizer.randomize(hum)