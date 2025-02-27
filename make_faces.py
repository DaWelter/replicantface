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

cam = bpy.context.scene.objects['Camera']

ENABLE_EXPENSIVE_RANDOMIZATION = True

def get_hairstyles():
    root = Path(bpy.app.binary_path).parent / 'humgen-assets'
    hair_folder = root / 'hair' / 'head'
    return [ str(p.relative_to(root)) for p in hair_folder.rglob('*.json') ]

HEAD_HAIRSTYLES = get_hairstyles()

# from https://polyhaven.com/hdris/skies/overcast
# and https://ambientcg.com
ENVS = list((Path(bpy.data.filepath).parent / 'hdris').glob('*.exr'))
assert len(ENVS) > 10


POSES = [
    'Standing around/HG_standing_looking_up.blend',
    'Standing around/HG_standing_pick_up_from_floor.blend',
    'Standing around/HG_standing_blocking_sun.blend',
    'Standing around/HG_Standing_looking_at_watch.blend',
    'Standing around/HG_Standing_waving.blend',
    'Standing around/HG_Standing_looking_back.blend',
    'Standing around/HG_standing_talking_on_phone.blend',
    'Sporting/HG_dumbbell_shoulder_press.blend',
    'Sporting/HG_basketball_jump.blend',
    'Sporting/HG_Stretch_quad.blend',
    #'Sporting/HG_Stretch_hamstring.blend',
    #'Sporting/HG_plank.blend',
    'Sporting/HG_football_kick.blend',
    'Sporting/HG_Yoga_pose.blend',
    'Sporting/HG_squat.blend',
    'Socializing/HG_raised_double_fist.blend',
    'Socializing/HG_Standing_arm_out_direction.blend',
    'Socializing/HG_waveing.blend',
    'Socializing/HG_Standing_handshake.blend',
    'Socializing/HG_gesturing.blend',
    'Socializing/HG_Standing_waving.blend',
    'Socializing/HG_thumbs_up.blend',
    'Socializing/HG_ecstatic.blend',
    'Socializing/HG_Standing_arms_out.blend',
    'Socializing/HG_standing_talking_on_phone.blend',
    'Socializing/HG_standing_arms_out_front.blend',
    'Sitting/HG_sitting_straight.blend',
    'Sitting/HG_sitting_body_forward.blend',
    'Sitting/HG_sitting_legs_together.blend',
    'Sitting/HG_sitting_legs_together_2.blend',
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


def randomize_background():
    filename = str(random.choice(ENVS))
    bpy.context.scene.world.node_tree.nodes["Environment Texture"].image = \
        bpy.data.images.load(filename)    


def randomize_human(hum_object : bpy.types.Object):
    hum : Human = Human.from_existing(hum_object)
    headbone = hum_object.pose.bones['head']

    if 1: # Pose
        new_pose = 'poses/'+random.choice(POSES)
        print('Pose = ',new_pose)
        hum.pose.set(new_pose, context = bpy.context)

    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Clothes
        hum.clothing.outfit.set_random(context = bpy.context)

    bpy.context.view_layer.update()

    if 1: # Rotation & Camera Position
        # Camera parameters
        r = random.randint(0, 2)
        if r == 0:
            distance = 0.8
            cam.data.lens = 70.
        elif r == 1:
            cam.data.lens = 200.
            distance = 2.2
        else:
            cam.data.lens = 35
            distance = 0.4
        distance *= 1.2
        cam.data.dof.focus_distance = distance
        
        hum_heading = random.uniform(0.,2.*pi)
        hum_rot = Euler((0., 0., hum_heading), 'XYZ')
        hum_object.rotation_euler = hum_rot
        # Manually update headbone to world matrix. Otherwise needs scene update.
        headbone_model_world : Matrix = hum_rot.to_matrix().to_4x4()
        headbone_model_world.translation = hum_object.matrix_world.translation
        headbone_model_world = headbone_model_world @ headbone.matrix

        r1 = random.uniform(-pi/2., pi/2.)
        r2 = random.uniform(-0.2*pi, 0.2*pi)
        wiggle_x = Euler((r2, 0., 0.), 'XYZ').to_quaternion()
        wiggle_z = Euler((0.,r1, 0.),'XYZ').to_quaternion()
        camera_position = headbone_model_world @ (wiggle_z @ wiggle_x @ Vector((0.,0.05,distance)))
        camera_rotation = headbone_model_world.to_3x3() @ wiggle_z.to_matrix() @ wiggle_x.to_matrix() # @ Euler((0.,np.pi,0.),'XYZ').to_matrix()

        cam.location = camera_position
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = camera_rotation.to_quaternion()

    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Body
        hum.body.randomize(context = bpy.context)

    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Face shape and age
        hum.face.randomize(context = bpy.context)
        hum.age.set(random.randint(10,60))

    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Skin and texture
        if hum.gender == 'male':
            texture_idx = random.randint(1,10)
            hum.skin.texture.set(f'textures/male/Default 1K/Male {texture_idx:02}.png')
        else:
            texture_idx = random.randint(1,10)
            hum.skin.texture.set(f'textures/female/Default 4K/Female {texture_idx:02}.png')
        hum.skin.roughness_multiplier.value = max(0.1, min(5., random.normalvariate(1.5,0.5)))
        hum.skin.normal_strength = random.uniform(1.,10)
        hum.skin.saturation = random.uniform(0.,2.)
        hum.skin.redness = random.uniform(-1.,1.)
        hum.skin.tone = random.uniform(0.2,3.)
        hum.skin.freckles = random.uniform(0.,0.5)
        hum.skin.splotches = random.uniform(0.,0.5)
        if hum.gender == 'female':
            for i in [2,4,6,8,10]:
                hum.skin.nodes['Gender_Group'].inputs[i].default_value = random.uniform(0.,1.)
        

    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Expression
        keys = hum.expression.keys
        assert len(keys) == 33
        idx = random.randint(0,len(keys)-1)
        for k in keys:
            k.value = 0.
        keys[idx].value = 1.
        # Close eyes with some probability
        eye1_ctrl, eye2_ctrl = [ k for k in hum.expression.keys if k.name.startswith('Blink') ]
        eye1_ctrl.value = 1. if random.randint(0,10)==0 else 0.
        eye2_ctrl.value = 1. if random.randint(0,10)==0 else 0.

    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Eyes
        hum.eyes.randomize()
        
    if 1 and ENABLE_EXPENSIVE_RANDOMIZATION: # Hair
        lightness = random.uniform(0.,4.)
        redness = random.uniform(0.,1.)
        snp = random.uniform(0.,1.)
        snp = snp**4
        hum.hair.regular_hair.set(random.choice(HEAD_HAIRSTYLES), context = bpy.context)
        hum.hair.regular_hair.lightness.value = lightness
        hum.hair.regular_hair.redness.value = redness
        hum.hair.regular_hair.salt_and_pepper.value = snp
        
        hum.hair.eyebrows.lightness.value = lightness
        hum.hair.eyebrows.redness.value = redness
        
        if hum.gender == 'male':
            hum.hair.face_hair.set_random(context = bpy.context)
            hum.hair.face_hair.lightness.value = lightness
            hum.hair.face_hair.redness.value = redness
            hum.hair.face_hair.salt_and_pepper.value = snp

    # Delete unwanted objects. The pirate outfit has a hat. It's the only one with
    # accessoirs which are out of scope for now so the hat is deleted.
    to_remove = [ o for o in bpy.context.scene.objects if 'hat' in o.name.lower() ]
    with bpy.context.temp_override(selected_objects = to_remove):
        bpy.ops.object.delete()


the_female = bpy.context.scene.objects['HG_Elenor']
the_male = bpy.context.scene.objects['HG_John']
assert is_part_of_human(the_female)
assert is_part_of_human(the_male)


def randomize_scene():
    print ("Randomizing")
    r = random.randint(0,1)
    for idx, hum_obj in enumerate([ the_female, the_male ]):
        objs = [ hum_obj ] + hum_obj.children_recursive
        for obj in objs:
            obj.hide_viewport = r!=idx
            obj.hide_render = r!=idx
    randomize_background()
    randomize_human([ the_female, the_male ][r])

def matrix_to_list(m):
    return list(list(col) for col in m)


@functools.lru_cache()
def get_face_vertex_indices():
    f = np.load(join(dirname(bpy.data.filepath),'head_indices.npz'))
    try:
        return f['indices']
    finally:
        f.close()


def get_face_vertices(human_obj) -> list[tuple[float,float,float]]:
    '''Vertices wrt headbone coord sys.'''
    headbone = human_obj.pose.bones['head']
    headbone_model_world = human_obj.matrix_world @ headbone.matrix
    c, = [ o for o in human_obj.children if 'Body' in o.name ]
    dg = bpy.context.evaluated_depsgraph_get()
    evaled = c.evaluated_get(dg)
    vert_trafo = headbone_model_world.inverted() @ evaled.matrix_world
    indices = get_face_vertex_indices()
    return [
        tuple(vert_trafo @ evaled.data.vertices[i].co) for i in indices
    ]

def debug_create_face_vertex_object(human_obj, vertices):
    headbone = human_obj.pose.bones['head']
    headbone_model_world = human_obj.matrix_world @ headbone.matrix
    m = bpy.data.meshes.new('TEST mesh')
    m.from_pydata(vertices, [], [])
    cpy = bpy.data.objects.new(name = 'TEST', object_data=m)
    cpy.matrix_world = headbone_model_world
    bpy.context.scene.collection.objects.link(cpy)
    bpy.context.view_layer.update()



def export_face_params(i : int, destination_dir : Path):
    hum_object, = [ o for o in [ the_female, the_male ] if not o.hide_render ]
    headbone = hum_object.pose.bones['head']
    m_world = hum_object.matrix_world @ headbone.matrix
    m_cam = cam.matrix_world.inverted() @ m_world
    render = bpy.context.scene.render
    proj_matrix = cam.calc_matrix_camera(
        bpy.context.view_layer.depsgraph,
        x = render.resolution_x,
        y = render.resolution_y,
        scale_x = render.pixel_aspect_x,
        scale_y = render.pixel_aspect_y)
    vertices = get_face_vertices(hum_object)
    m_cam = matrix_to_list(m_cam)
    proj_matrix = matrix_to_list(proj_matrix)
    filename = destination_dir / f'face{i:05}.npz'
    np.savez_compressed(filename,
                        modelview = np.asarray(m_cam, dtype=np.float32),
                        projection = np.asarray(proj_matrix, dtype=np.float32),
                        vertices = np.asarray(vertices, dtype=np.float16),
                        resolution = render.resolution_x)
    
    

def render_and_save(i : int, destination_dir : Path):
    bpy.context.scene.render.filepath = str(destination_dir / f'face{i:05}.jpg')
    print ("Rendering to ", bpy.context.scene.render.filepath)
    bpy.ops.render.render(animation=False, write_still=True)


argv = sys.argv
argv = list(itertools.dropwhile(lambda x: x != '--', argv))
if argv:
    assert argv[0] == '--'
    argv = argv[1:]

if not bpy.app.background:
    randomize_scene()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--count", default=1, type=int)
    parser.add_argument("-o", type=str)
    parser.add_argument("--cycles-device", help="dummy")
    args = parser.parse_args(argv)
    output_dir = Path(args.o)
    start : int = args.start
    count : int = args.count
    print (f"Rendering {count} faces starting from index {start} to directory {output_dir}")
    for i in range(start, start+count):
        randomize_scene()
        bpy.context.view_layer.update()
        render_and_save(i, output_dir)
        export_face_params(i, output_dir)