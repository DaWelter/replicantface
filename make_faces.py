"""Entry point for random face generation."""
import bpy
from math import pi
from pathlib import Path
from HumGen3D import Human
import random
import argparse
import sys
import itertools
import sys
import shutil
import numpy as np
import os

if __name__ == '__main__':
    # Trigger reimport when the script is run again.
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]


from replicantface import (export_face_params, 
                           randomize_body_pose, 
                           randomize_expression,
                           hide_object, 
                           find_hum, 
                           Accessoires, 
                           Hair, 
                           EnvRandomizer, 
                           sample_ethnicity, 
                           ETHNICITIES, 
                           randomize_face_shape, 
                           randomize_body_shape, 
                           sample_texture, 
                           randomize_skin_color,
                           update_child_of_constraint,
                           ClothesRandomizer,
                           setup_extra_face_material_selection, 
                           update_compositing,
                           fix_shapekeys,
                           randomize_camera_parameters,
                           HeadCoverage,
                           sample_pose,
                           PoseSample)


class Randomizer:
    def __init__(self):
        scene = bpy.data.scenes['Scene']
        bpy.context.window.scene = scene
        self.cam = scene.objects['Camera']
        self._accessoires = Accessoires(p_glasses=0.2,p_hat=0.1,p_phones=0.02, p_mask=0.02, root=scene.objects['accessoires'])
        self._hair = Hair()
        self._env = EnvRandomizer(p_custom_light=0.05)
        self._clothes = ClothesRandomizer(p_naked=0.01, p_pattern=0.2)
        self._hum = self._get_human()


    def _get_human(self):
        try:
            return find_hum()
        except RuntimeError:
            pass

        # Create one
        r = random.randint(0,1)
        if r == 0:
            hum = Human.from_preset('models/male/Caucasian Presets/Caucasian 1.json', bpy.context)
        else:
            hum = Human.from_preset('models/female/Caucasian presets/Caucasian 1.json', bpy.context)
        fix_shapekeys(hum)
        return hum


    def _randomize_human(self):
        hum = self._hum
        age = random.randint(20,70)
        if 1:
            hum.age.set(age)
            eth = sample_ethnicity()

        if 1: # Face
            randomize_face_shape(hum, age, eth)
            sample_texture(hum, eth, p_uglified=0.5)
            randomize_skin_color(hum)

        if 1: # Body shape
            randomize_body_shape(hum)

        if 1: # Pose
            randomize_body_pose(hum)

        if 1: # Accessoires
            update_child_of_constraint(self._accessoires.root, hum.objects.rig, 'head')
            head_covered = self._accessoires.enable_randomly()
            self._accessoires.randomize_materials()
            bpy.context.view_layer.update()

        if 1: # Hair
            self._hair.randomize(hum, age, head_covered)

        if 1: # Clothes
            self._clothes.randomize(hum)


    def _randomize_expressions(self):
        hum = self._hum
        if 1: # Expression
            randomize_expression(hum, p_neutral=0.2, p_eyes_closed=0.05, p_open_mouth=0.2)
        if 1: # Eyes
            hum.eyes.randomize()        


    def randomize(self,*, new_human : bool, expression_and_env : bool):
        hum = self._hum
        hum_obj = hum.objects.rig
        if new_human:
            sample_pose(wide_distribution=True).apply_to_scene(self.cam, self._hum)
            self._randomize_human()
        if expression_and_env:
            self._randomize_expressions()
            randomize_camera_parameters(self.cam, bpy.data.scenes['EnvScene'].objects['Camera2'])
            update_child_of_constraint(self._env.rotation_helper, hum_obj, 'head')
            self._env.randomize()
        bpy.context.view_layer.update()
        return hum, hum_obj, self.cam


def render_and_save(image_prefix_path : Path):
    assert bpy.context.scene.frame_current == 0
    bpy.ops.render.render(animation=False, write_still=True, scene="EnvScene")
    bpy.ops.render.render(animation=False, write_still=True, scene="Scene")
    # Fixing trailing zeros from compositing output
    p = image_prefix_path
    composit_fn = p.with_name(p.stem + '_mask0000.png')
    fixed_fn = p.with_name(p.stem + '_mask.png')
    shutil.move(composit_fn, fixed_fn)
    composit_fn = p.with_name(p.stem + '_image0000.jpg')
    fixed_fn = p.with_name(p.stem + '_img.jpg')
    shutil.move(composit_fn, fixed_fn)
    os.unlink(p)


if __name__ == '__main__':
    argv = sys.argv
    argv = list(itertools.dropwhile(lambda x: x != '--', argv))
    if argv:
        assert argv[0] == '--'
        argv = argv[1:]  
    if not bpy.app.background: # Can be run in the editor, too.
        randomizer = Randomizer()
        _, hum_obj, _ = randomizer.randomize(new_human=True, expression_and_env=False)
        setup_extra_face_material_selection(hum_obj)
        update_compositing(bpy.context.scene)
    else: 
        # Headless mode
        parser = argparse.ArgumentParser()
        partial_gen_group = parser.add_mutually_exclusive_group()
        partial_gen_group.add_argument("--dump-new-human", type=str, default='', 
                                       help="Writes the scene as blender file to the given filename")
        partial_gen_group.add_argument("--randomize-existing", action="store_true", default=False,
                                       help="Assumes a human exists and only varies facial expressions, env and so on.")
        parser.add_argument("--cycles-device", help="dummy")
        args = parser.parse_args(argv)
        image_prefix_path = Path(bpy.context.scene.render.filepath)
        randomizer = Randomizer()
        if args.randomize_existing:
            _, hum_obj, cam = randomizer.randomize(new_human=False, expression_and_env=True)
        elif args.dump_new_human:
            _, hum_obj, cam = randomizer.randomize(new_human=True, expression_and_env=False)
            bpy.ops.wm.save_mainfile(filepath=args.dump_new_human, check_existing=False)
        else:
            _, hum_obj, cam = randomizer.randomize(new_human=True, expression_and_env=True)
        setup_extra_face_material_selection(hum_obj)
        update_compositing(bpy.context.scene)
        render_and_save(image_prefix_path)
        export_face_params(hum_obj, cam)