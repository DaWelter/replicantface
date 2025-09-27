import bpy
from math import pi
from os.path import join, dirname, basename, splitext
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
from pathlib import Path

from HumGen3D import Human, HumGenException


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


def setup_extra_face_material_selection(hum_obj : bpy.types.Object):
    '''In order to composit masks by materials.'''
    body, = [ o for o in hum_obj.children if 'body' in o.name.lower() ]

    def clone_human_mat(name):
        if name not in bpy.data.materials:    
            mat_copy = bpy.data.materials['.Human'].copy()
            mat_copy.name = name
        else:
            print (f"Using existing material {name}")
            mat_copy = bpy.data.materials[name]
        if not any(m.name==name for m in body.data.materials):
            body.data.materials.append(mat_copy)
        else:
            print (f"Material {name} is already assigned")

    def assign_material(indices, mat_name):
        indices = set(indices)
        mat_idx, = [i for i,m in enumerate(body.data.materials) if m.name == mat_name]
        for i, p in enumerate(body.data.polygons):
            sel = i in indices
            p.select = sel
            if sel:
                p.material_index = mat_idx

    clone_human_mat('.Face')
    clone_human_mat('.Ears')

    indices = np.load(join(dirname(bpy.data.filepath),'skull_polygon_indices.npz'))['indices']
    assign_material(indices, '.Face')
    indices = np.load(join(dirname(bpy.data.filepath),'ears_polygon_indices.npz'))['indices']
    assign_material(indices, '.Ears')


def update_compositing(scene : bpy.types.Scene):
    '''Adjust compositing nodes
    
    - Assigns material names
    - Sets file paths
    '''

    body_materials = set()
    face_materials = set()
    accessoire_materials = set()
    beard_materials = set()
    hair_materials = set()
    materials = {
        m for o in scene.objects for m in getattr(o.data, 'materials', [])
    }
    for m in materials:
        if m.name.startswith('accessoire_'):
            accessoire_materials.add(m.name)
        elif m.name.startswith('.Human'):
            body_materials.add(m.name)
        elif m.name.startswith('.Face'):
            face_materials.add(m.name)
        elif m.name.startswith('.HG_Hair_Eye') or m.name.startswith('.HG_Eyes') or m.name.startswith('.HG_Teeth'):
            face_materials.add(m.name)
        elif m.name.startswith('.HG_Hair_Face'):
            beard_materials.add(m.name)
        elif m.name.startswith('.HG_Hair_Head'):
            hair_materials.add(m.name)
        #elif m.name.startswith('.HG_'):
        else:
            # Anything not already covered, is probably clothing.
            body_materials.add(m.name)

    scene.node_tree.nodes["CryptomatteBody"].matte_id = ','.join(body_materials)
    scene.node_tree.nodes["CryptomatteAccessoires"].matte_id = ','.join(accessoire_materials)
    scene.node_tree.nodes["CryptomatteFace"].matte_id = ','.join(face_materials)
    scene.node_tree.nodes["CryptomatteNeck"].matte_id = '.Ears'
    scene.node_tree.nodes["CryptomatteHair"].matte_id = ','.join(hair_materials)
    scene.node_tree.nodes["CryptomatteBeard"].matte_id = ','.join(beard_materials)

    image_filename = bpy.context.scene.render.filepath
    scene.node_tree.nodes["MaskOutput"].base_path = dirname(image_filename)
    scene.node_tree.nodes["MaskOutput"].file_slots[0].path = splitext(basename(image_filename))[0]+'_mask'
    scene.node_tree.nodes['ImageOutput'].base_path = dirname(image_filename)
    scene.node_tree.nodes["ImageOutput"].file_slots[0].path = splitext(basename(image_filename))[0]+'_image'


def export_face_params(hum_object : bpy.types.Object, cam : bpy.types.Object):
    image_path = Path(bpy.context.scene.render.filepath)
    label_path = image_path.with_suffix('.npz')
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
    vertices = get_face_vertices(hum_object) # Only the vertices of the head are saved.
    m_cam = matrix_to_list(m_cam)
    proj_matrix = matrix_to_list(proj_matrix)
    np.savez_compressed(str(label_path),
                        modelview = np.asarray(m_cam, dtype=np.float32),
                        projection = np.asarray(proj_matrix, dtype=np.float32),
                        vertices = np.asarray(vertices, dtype=np.float16),
                        resolution = render.resolution_x)


if __name__ == '__main__':
    obj = bpy.context.object
    setup_extra_face_material_selection(obj)
    update_compositing(bpy.context.scene)