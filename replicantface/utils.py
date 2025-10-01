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
import os
import shutil
from functools import partial, lru_cache
from typing import Any, NamedTuple, Literal
import math
from pathlib import Path
import copy
from HumGen3D import Human


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


def hide_object(obj : bpy.types.Object, val : bool):
    obj.hide_set(val)
    obj.hide_render = val


def gaussian(x, mu, sigma):
    return math.exp(-((x - mu)/sigma)**2)


def update_child_of_constraint(obj : bpy.types.Object, target : bpy.types.Object, subtarget : str):
    c = obj.constraints['Child Of']
    if c.subtarget != subtarget or c.target is not target:
        c.target = target
        c.subtarget = subtarget


def random_beta_11(concentration):
    return 2.*(random.betavariate(concentration,concentration) - 0.5)


@lru_cache
def replicantface_folder() -> Path:
    candidates = [
        Path(bpy.data.filepath).parent,
        Path(os.getcwd()),
    ]
    for path in candidates:
        if (path / 'replicantface').is_dir() and (path / 'hdris').is_dir() and (path / 'assets').is_dir():
            return path
    raise RuntimeError("Cannot find replicantface folder")


HeadTopCoverage = Literal['none','loose','tight']

class HeadCoverage(NamedTuple):
    top_covered : HeadTopCoverage
    allow_beard : bool