import bpy
from math import pi
from os.path import join, dirname
from pathlib import Path
from mathutils import Euler, Vector, Matrix
from HumGen3D import Human, HumGenException
from HumGen3D.common import is_part_of_human
import random
import json
import argparse
import sys
import itertools
import functools
import numpy as np
import numpy.typing as npt
import os
import shutil
from functools import partial
from typing import Any, Literal
import math
from PIL import Image

if __name__ == '__main__':
    # Trigger reimport when the script is run again.
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]

from replicantface.utils import replicantface_folder


def smooth_coin_flip(minval, maxval):
    # The middle range from 0.1 to 0.9 has 30% probability to find a value in there. The other 60% of the probability
    # mass is split off on the sides.
    # return random.betavariate(0.3,0.3)*(maxval-minval) + minval

    if random.randint(0,1) == 0:
        return random.betavariate(4.,11.)*(maxval-minval) + minval
    else:
        return random.betavariate(11.,4.)*(maxval-minval) + minval

def smooth_coin_flip_22():
    #return random.uniform(-2., 2.)
    return smooth_coin_flip(-2., 2.)


def smooth_coin_flip_01():
    #return random.uniform(0., 1.)
    return random.betavariate(0.3,0.3)

def l1_limit(x, max_norm=1.):
    norm = np.linalg.norm(x, ord=1)
    if norm > max_norm:
        return x * max_norm / norm
    return x


@functools.lru_cache(maxsize=1)
def get_uglification_layer(gender : Literal['male','female']) -> Image.Image:
    return Image.open(replicantface_folder() / 'assets' / f'{gender}_uglification_layer.png')


def set_texture(hum : Human, texture_idx : int, gender : Literal['male','female'], uglified: bool):
    root = Path(bpy.app.binary_path).parent / 'humgen-assets'
    texture_file = Path("textures") / gender / "Default 4K" / f"{gender.capitalize()} {texture_idx:02}.png"
    if uglified:
        uglifier = np.asarray(get_uglification_layer(gender)).astype(np.uint16)
        source = np.asarray(Image.open(root / texture_file).convert('RGB')).astype(np.uint16)
        img = Image.fromarray((source * uglifier / 256).astype(np.uint8))
        texture_file = Path("textures") / gender / "Default 4K" / 'tmp_uglified_face.png'
        # Caution: PNG saves very slowly. Jpeg is an order of magnitude faster!
        img.save(root / texture_file, 'jpeg',quality=90)
        for imdata in bpy.data.images:
            if Path(imdata.filepath).name != texture_file.name:
                continue
            imdata.reload()
    hum.skin.texture.set(str(texture_file))


def sample_triangle():
    u1 = random.uniform(0., 1.)
    u2 = random.uniform(0., 1.)
    if u1+u2 > 1.:
        u1 = 1. - u1
        u2 = 1. - u2
    u3 = 1. - u1 - u2
    return u1, u2, u3

def sample_body_proportion_weights():
    # Fat, Muscle, Skinny
    presets = np.asarray([
        (1.  , 1. , -0.7), # Fat, Muscular
        (1.  , 1. ,  0. ), # Heavy Muscular
        (1.  ,-0.5,-0.5), # Really Fat
        (1.  , 0. ,  0. ),
        (1.  , 0. , -0.5),
        (0.  , 1. , -0.5), # Schwarzenegger
        (0.  , 1. ,  0. ), # Muscular
        (0.  , 1. ,  0.7), # Skinny Muscular
        (0.  , 0. ,  1. ), # Skinny
        (0.  , 0. , -0.5), # Not Skinny
        (-0.5, 0. ,  1. ), # Starving
        (-0.5,-0.5,  1. ), # Starving
    ])
    if random.randint(0,1) == 0:
        us = np.zeros((3,), dtype=np.float32)    
    else:
        us = random.choice(presets)
    us = us + 0.1*np.random.uniform(-1., 1., size=3)
    us = np.clip(us, -1., 1.)
    return us



ETHNICITIES = ['black', 'caucasian', 'asian']


def sample_ethnicity() -> dict[str,float]:
    weights = {}
    noises = (np.random.beta(4.,4.,size=3)-0.5)*0.2
    for noise, name in zip(noises,ETHNICITIES):
        w = {
            0 : -0.5,
            1 : 0.,
            2 : 0.9
        }[random.randint(0,2)]
        weights[name] = w + noise
    return weights


def _set_key(key, new_value):
    print ("Set ", key.name, " to ", new_value)
    if hasattr(key, "set_without_update"):
        key.set_without_update(new_value)
    else:
        key.value = new_value


def randomize_face_shape(hum : Human, age : float, ethnicity_weights : dict[str,float]):
    # eyes', 'l_skull', 'ears', 'special', 'chin', 'cheeks', 'mouth', 'jaw', 'nose', 'u_skull', None

        # 'eye_orbit_size', 
        # 'eye_tilt', 
        # 'eyelid_shift_horizontal', 
        # 'eyelid_rotation', 
        # 'eye_width', 
        # 'eyelid_fat_pad', 
        # 'Eye Height', 
        # 'Eye Distance', 
        # 'eyelid_shift_vertical', 
        # 'eye_height', 
        # 'Eye Depth', 
        # 'nose_tip_width', 
        # 'nose_nostril_turn', 
        # 'nose_location', 
        # 'nose_height', 
        # 'nose_tip_angle', 
        # 'nose_bridge_height', 
        # 'nose_tip_length', 
        # 'nose_angle', 
        # 'nose_bridge_width', 
        # 'nose_nostril_flare', 
        # 'nose_tip_size', 
        # 'lip_cupid_bow', 
        # 'lip_offset', 
        # 'lip_width', 
        # 'lip_location', 'lip_height', 'cheek_zygomatic_bone', 'cheek_zygomatic_proc', 'cheek_fullness', 'ear_width', 'ear_turn', 'ear_height', 'ear_lobe_size', 'ear_antihelix_shape', 'muzzle_location_vertical', 'muzzle_location_horizontal', 'browridge_center_size', 'forehead_size', 'browridge_loc_vertical', 'browridge_loc_horizontal', 'temple_size', 'chin_dimple', 'chin_size', 'chin_height', 'chin_width', 'jaw_location_vertical', 'jaw_location_horizontal', 'jaw_width', 'Eye Scale'])

    keys_by_name = { k.name:k for k in hum.keys.filtered("face_presets") }

    # First select ethnicity
    ethnicity_keys = { n:keys_by_name.pop(n) for n in ETHNICITIES }
    for name, weight in ethnicity_weights.items():
        _set_key(ethnicity_keys[name], weight)

    # Then mix in additional presets
    keys = np.asarray(list(keys_by_name.values()),dtype=object)
    del keys_by_name
    weights = np.zeros((len(keys),), dtype=np.float32)

    num_select = random.randint(0, 4)
    normalizers = {
        1 : 1.,
        2 : 1.,
        3.: 2.,
        4.: 4.
    }
    neg_scales = {
        1 : -0.9,
        2 : -0.5,
        3.: -0.5,
        4.: -0.25
    }
    pos_scales = {
        1 : 0.9,
        2 : 0.9,
        3.: 0.5,
        4.: 0.5
    }

    for i in np.random.choice(len(keys), size=(num_select,), replace=False):
        weights[i] = {
            0 : neg_scales[num_select],
            1 : pos_scales[num_select]
        }[random.randint(0,1)] # / normalizers[num_select]

    weights += l1_limit((np.random.beta(3.,3.,size=len(keys))-0.5)*0.2, max_norm=0.1)

    for key, wn in zip(keys, weights):
        _set_key(key, float(wn))

    # Add details
    if 1:
        keys_by_name = { k.name:k for k in hum.keys.filtered("face_proportions") }
        _set_key(keys_by_name.pop('Eye Height'), random.normalvariate(0., 0.1))
        _set_key(keys_by_name.pop('Eye Distance'), random.normalvariate(0., 0.1))
        _set_key(keys_by_name.pop('Eye Depth'), random.normalvariate(0., 0.1))
        _set_key(keys_by_name.pop('Eye Scale'), random.normalvariate(0.,0.2))
        # print ("Face proportion names: ", keys_by_name.keys())
        # _set_key(keys_by_name.pop('forehead_size'), smooth_coin_flip_22())
        # _set_key(keys_by_name.pop('temple_size'), smooth_coin_flip_22())
        # _set_key(keys_by_name.pop('ear_width'), smooth_coin_flip_22())
        # _set_key(keys_by_name.pop('jaw_width'), smooth_coin_flip_22())
        # _set_key(keys_by_name.pop('ear_turn'), smooth_coin_flip_01())
        # weight_noise = l1_limit(np.random.normal(0., 0.3, size=len(keys_by_name)), max_norm=5.)
        # for w, (name, key) in zip(weight_noise, keys_by_name.items()):
        #     _set_key(key, w)
        range_overrides = {
            'nose_tip_width' : (-1., 1.),
            'nose_location' : (-1.,1.),
            'nose_tip_size' : (-1.,1.)
        }
        weights = {
            n:0. for n in keys_by_name.keys()
        }
        num_changed = random.randint(0, len(keys_by_name)-1)
        for n in np.random.choice(list(keys_by_name.keys()), size=num_changed, replace=False):
            weights[n] = smooth_coin_flip(*range_overrides.get(n,(-2.,2.)))
        
        for n, w in weights.items():
            _set_key(keys_by_name[n], w)

    hum.keys.update_human_from_key_change(bpy.context)


def randomize_body_shape(hum : Human):
    #hum.body.randomize(context = bpy.context)
    keys_by_name = { k.name:k for k in hum.keys.filtered("body_proportions") }
    _set_key(keys_by_name.pop('Neck Thickness'), random.normalvariate(0., 0.1))
    _set_key(keys_by_name.pop('Neck Length'), random.uniform(-0.5,0.0))
    u1,u2,u3 = sample_body_proportion_weights()
    _set_key(keys_by_name.pop('overweight'), u1)
    _set_key(keys_by_name.pop('muscular'), u2)
    _set_key(keys_by_name.pop('skinny'), u3)
    hum.keys.update_human_from_key_change(bpy.context)


def sample_texture(hum : Human, ethnicity_weights : dict[str,float], p_uglified : float):
    N = 10 # textures
    texture_weights = np.asarray([
        np.linspace(0.,1.,N),
        np.linspace(1.,0.,N),
        np.ones((N,))
    ])
    ethnicity_weights = np.asarray([
        ethnicity_weights[k] for k in ETHNICITIES
    ])
    ethnicity_weights /= np.linalg.norm(ethnicity_weights,ord=1)
    texture_weights = np.sum(texture_weights * ethnicity_weights[:,None],axis=0) + np.ones((N,))
    texture_weights = np.maximum(0., texture_weights)
    texture_weights /= np.sum(texture_weights)
    
    texture_idx = int(np.random.choice(N,p=texture_weights)+1)
    set_texture(hum, texture_idx, hum.gender, uglified=np.random.uniform(0.,1.)<p_uglified)


def randomize_skin_color(hum : Human, p_makeup = 0.5):
    hum.skin.set_subsurface_scattering(True,bpy.context)
    hum.skin.randomize()
    hum.skin.roughness_multiplier.value = max(0.1, min(5., random.normalvariate(1.5,0.5)))
    # Not working?!
    # hum.skin.normal_strength = random.uniform(1.,10)
    # hum.skin.saturation = random.uniform(0.,2.)
    # hum.skin.redness = random.uniform(-1.,1.)
    # hum.skin.tone = random.uniform(0.2,3.)
    # hum.skin.freckles = random.uniform(0.,0.5)
    # hum.skin.splotches = random.uniform(0.,0.5)
    # Makeup?
    BLUSH=4
    EYESHADOW=6
    LIPSTICK=8
    EYELINER=10
    def set_makeup(i : int, opacity : float, color : tuple[float,float,float]):
        hum.skin.nodes['Gender_Group'].inputs[i].default_value = opacity
        hum.skin.nodes['Gender_Group'].inputs[i+1].default_value = (*color, 1.)

    if hum.gender == 'female':
        makeup_enable = random.uniform(0.,1.) < p_makeup
        set_makeup(BLUSH,
                   smooth_coin_flip_01() * makeup_enable,
                   random.choice([
                       (0.8,0.6,0.5),
                       (1.0,0.,0.)
                   ]))
        set_makeup(EYESHADOW,
                   smooth_coin_flip_01() * makeup_enable,
                   random.choice([
                       (0.4,0.1,0.1),
                       (0.1,0.2,0.25),
                       (0.01,0.01,0.02),
                   ]))
        set_makeup(LIPSTICK,
                   smooth_coin_flip_01() * makeup_enable,
                   random.choice([
                       (0.3,0.09,0.07),
                       (0.5,0.0,0.0),
                       (0.,0.,0.),
                       (0.65,0.13,0.19)
                   ]))
        set_makeup(EYELINER,
                   smooth_coin_flip_01() * makeup_enable,
                   (0.,0.,0.))
        # "Foundation" basically paints the entire face in some colors. Pretty useless.
        hum.skin.nodes['Gender_Group'].inputs[2].default_value = 0.
    hum.keys.update_human_from_key_change(bpy.context)


if __name__ == '__main__':
    def find_hum(gender : str | None = None) -> Human:
        for obj in bpy.context.scene.objects:
            if not obj.name.startswith('HG_'):
                continue
            try:
                hum : Human = Human.from_existing(obj)
            except HumGenException:
                continue
            if gender and hum.gender != gender:
                continue
            return hum
        raise RuntimeError("Human not found")
    hum = find_hum()
    eth = sample_ethnicity()
    #randomize_face_shape(hum, 0., eth)
    #randomize_body_shape(hum)
    #sample_texture(hum, eth, 0.5)
    randomize_skin_color(hum)