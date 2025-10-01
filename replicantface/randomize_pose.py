import dataclasses
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
from functools import partial
from typing import Any
import math
from numpy.typing import NDArray


if __name__ == '__main__':
    # Trigger reimport when the script is run again.
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]

from replicantface.export import get_face_vertex_indices
from replicantface.utils import update_child_of_constraint, random_beta_11


# Not used because of occluding face easily
POSES_POTENTIALLY_OCCLUDING = [
    'Standing around/HG_standing_blocking_sun.blend',
    'Sporting/HG_basketball_jump.blend',
    'Sporting/HG_dumbbell_shoulder_press.blend'
    'Sitting/HG_sitting_legs_together_2.blend',
    'Standing around/HG_standing_talking_on_phone.blend',
    'Socializing/HG_raised_double_fist.blend',
    'Socializing/HG_standing_talking_on_phone.blend',
    'Socializing/HG_Standing_waving.blend',
    'Socializing/HG_waveing.blend',
]

POSES = [
    'Standing around/HG_standing_looking_up.blend',
    'Standing around/HG_standing_pick_up_from_floor.blend',
    'Standing around/HG_Standing_looking_at_watch.blend',
    'Standing around/HG_Standing_waving.blend',
    'Standing around/HG_Standing_looking_back.blend',
    
    'Sporting/HG_Stretch_quad.blend',
    #'Sporting/HG_Stretch_hamstring.blend',
    #'Sporting/HG_plank.blend',
    'Sporting/HG_football_kick.blend',
    'Sporting/HG_Yoga_pose.blend',
    'Sporting/HG_squat.blend',
    
    'Socializing/HG_Standing_arm_out_direction.blend',
    
    'Socializing/HG_Standing_handshake.blend',
    'Socializing/HG_gesturing.blend',
    
    'Socializing/HG_thumbs_up.blend',
    'Socializing/HG_ecstatic.blend',
    'Socializing/HG_Standing_arms_out.blend',
    
    'Socializing/HG_standing_arms_out_front.blend',
    'Sitting/HG_sitting_straight.blend',
    'Sitting/HG_sitting_body_forward.blend',
    'Sitting/HG_sitting_legs_together.blend',

    'Sitting/HG_sitting_straight_looking_at_phone.blend',
    'Sitting/HG_sitting_explaining.blend',
    'Sitting/HG_sitting_hands_together.blend',
    'Sitting/HG_sitting_lean_forward.blend',
    'Walking/HG_Walking.blend',
    'Walking/HG_walking_5.blend',
    'Walking/HG_walking_4.blend',
    'Walking/HG_walking_3.blend',
    'Walking/HG_walking_2.blend',
    'Walking/HG_standing_step_up.blend',
    'Walking/HG_walking_6.blend',
    'Walking/HG_standing_step_down.blend',
    'Running/HG_Running_4.blend',
    'Running/HG_Running_7.blend',
    'Running/HG_Running_5.blend',
    'Running/HG_Running_2.blend',
    'Running/HG_Running_3.blend',
    'Running/HG_Running_8.blend',
    'Running/HG_Running_1.blend',
    'Running/HG_Running_6.blend'
]


def fix_shapekeys(hum : Human):
    # A bunch of vertices from the lip deviate from the base shape.
    # This function fixes that by copying all head vertices to the broken blend shapes.
    # Otherwise the shapes don't affect the head.
    body = hum.objects.body
    indices = get_face_vertex_indices()
    basevertices = body.data.shape_keys.key_blocks['Basis'].data
    for name in ['cor_ElbowBend_Lt', 
                 'cor_ElbowBend_Rt',
                 'cor_ShoulderSideRaise_Lt',
                 'cor_ShoulderSideRaise_Rt',
                 'cor_ShoulderFrontRaise_Lt',
                 'cor_ShoulderFrontRaise_Rt',
                 'cor_LegFrontRaise_Lt',
                 'cor_LegFrontRaise_Rt',
                 'cor_FootDown_Lt',
                 'cor_FootDown_Rt']:
        vertices = body.data.shape_keys.key_blocks[name].data
        for i in indices:
            q = basevertices[i].co
            vertices[i].co = q


@dataclasses.dataclass
class PoseSample:
    head_heading : float
    head_pitch : float
    head_roll : float
    cam_heading : float
    cam_pitch : float
    cam_roll : float

    def apply_to_scene(self, cam : bpy.types.Object, hum : Human):
        body : bpy.types.Object = hum.objects.body
        rig : bpy.types.Object = hum.objects.rig

        # Camera parameters
        update_child_of_constraint(cam.parent, body, 'head')

        bones = rig.pose.bones
        headbone = bones['head']
        neckbone = bones['neck']
        for b in [ headbone, neckbone ]:
            b.rotation_mode = 'XYZ'
            b.rotation_euler = (self.head_pitch/2.,self.head_heading/2.,self.head_roll/2.)

        cam.parent.rotation_euler[2] = self.cam_heading
        cam.parent.rotation_euler[1] = self.cam_roll
        cam.parent.rotation_euler[0] = pi/2. + self.cam_pitch


def sample_pose(wide_distribution : bool = False):
    if wide_distribution:
        # Wider distributions because using uniform distributions.
        # For the camera pitch there is also a uniform range specified. But due to how rotations 
        # work, the actual pitch/roll distribution of the face in camera space will be "softened",
        # looking more like a gaussian.
        # To compensate, the camera could be rolled. I'd rather keep the horizon level though.
        heading = 70./180.*pi*random.uniform(-1.,1.)
        cam_heading = 80./180.*pi*random.uniform(-1.,1.) + heading
        cam_pitch = pi/180.*(-5.+40.*random.uniform(-1.,1.))
        while True:
            # Relative to the camera, as well as relative to the body (for realism) we want at 
            # most 40 deg pitch.
            pitch = 40./180.*pi*random.uniform(-1.,1.) + cam_pitch
            if abs(pitch) < 40./180.*pi:
                break
        roll = 30./180.*pi*random_beta_11(2.)
    else:
        heading = 70./180.*pi*random_beta_11(4.)
        pitch = 40./180.*pi*random_beta_11(4.)
        roll = 30./180.*pi*random_beta_11(4.)

        if False: #random.randint(0,100) == 0:
            cam_heading = random.uniform(-pi,pi)
        else:
            cam_heading = random_beta_11(4.)*90.*pi/180.
        cam_pitch = (-5. + random_beta_11(4.)*20.)*pi/180. # Relative to the world

    return PoseSample(
        head_heading=heading,
        head_pitch=pitch,
        head_roll=roll,
        cam_heading = cam_heading,
        cam_pitch = cam_pitch,
        cam_roll = 0.
    )


def randomize_body_pose(hum : Human):
    # Change body pose
    if 0:
        # Probably not needed. In FaceSynth, it looks like the mostly the head moves
        # and the rest of the body remains in default pose.
        new_pose = 'poses/'+random.choice(POSES)
        hum.pose.set(new_pose, context = bpy.context)
        print ("selected pose: ", new_pose)
    
    # change gaze direction
    rig : bpy.types.Object = hum.objects.rig
    eyetarget = rig.pose.bones['eyeball_lookat_master']
    eyetarget.location[0] = 0.1 * random_beta_11(3.)
    eyetarget.location[2] = 0.03 * random_beta_11(3.)


def randomize_camera_parameters(cam : bpy.types.Object, env_cam : bpy.types.Object):
    distance = cam.location[2]
    cam.data.dof.focus_distance = distance
    # Env cam
    env_cam.data.lens = random.choice([25.,30.,50.,70.])


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
    cam = bpy.context.scene.objects['Camera']
    hum = find_hum()
    fix_shapekeys(hum)
    randomize_body_pose(hum)
    randomize_camera_parameters(cam, bpy.data.scenes['EnvScene'].objects['Camera2'])

    pose_sample = sample_pose()
    pose_sample.apply_to_scene(cam, hum)