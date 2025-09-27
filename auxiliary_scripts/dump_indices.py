"""This script was used to extract vertex / polygon indices for various parts of the HumGen mesh."""

import bpy
import numpy as np
from os.path import join, dirname

DSTDIR = dirname(bpy.data.filepath)

if 0:
    # Output selected vertices.
    c = bpy.context.active_object
    indices = []
    for i,v in enumerate(c.data.vertices):
        if v.select:
            indices.append(i)
    print('selected num verts: ', len(indices))
    np.savez(join(DSTDIR,'face_indices.npz'), indices = np.asarray(indices,dtype=np.int64))


if 1:
    # Output selected polygons. Used for creating segmentation masks.
    # These selections were always created by eye measure.
    c = bpy.context.active_object
    indices = []
    for i,p in enumerate(c.data.polygons):
        if p.select:
            indices.append(i)
    print('selected num poly: ', len(indices))
    np.savez(join(DSTDIR,'ears_polygon_indices.npz'), indices = np.asarray(indices,dtype=np.int64))


if 0:
    # This part figures out the indices of the facial landmarks of the 68 point scheme.
    # Create a scene where the BFM facial mesh from 3DDFA_2 (https://github.com/cleardusk/3DDFA_V2/tree/master/configs)
    # is aligned with the face from HumGen. Then you can use this script to get the closest match on the
    # HumGen mesh.

    hum_obj = bpy.context.scene.objects['HG_Clay'] # Use the name of your human.
    head_obj = bpy.context.scene.objects['face'] # Face mesh from the BFM.
    headbone = hum_obj.pose.bones['head']
    headbone_model_world = hum_obj.matrix_world @ headbone.matrix
    translation = (headbone_model_world.inverted() @ head_obj.matrix_world).col[3]

    # Landmarks wrt the BFM mesh.
    landmark_indices = np.array([17440, 17716, 17222, 16770, 33981, 34802, 35355, 35777, 36091,
        36403, 36806, 37333, 38086, 29006, 28546, 28036, 28276, 29704,
        30401, 30791, 30996, 31163, 31902, 32068, 32276, 32670, 33376,
            8161,  8177,  8187,  8192,  6515,  7243,  8204,  9163,  9883,
            1959,  3887,  5048,  6216,  4674,  3513,  9956, 11223, 12384,
        14327, 12656, 11495,  5522,  6025,  7495,  8215,  8935, 10395,
        10795,  9555,  8836,  8236,  7636,  6915,  5909,  7384,  8223,
            9064, 10537,  8829,  8229,  7629])

    # Debug
    for v in head_obj.data.vertices:
        v.select = False
    for v in (head_obj.data.vertices[i] for i in landmark_indices):
        v.select = True

    # Get the vertices of the HumGen mesh. It must have deformers evaluated or the results won't be correct.
    # Works only in object mode. Else the eval'd object won't have vertices.
    body_obj = bpy.context.scene.objects['HG_Body']
    bpy.context.view_layer.objects.active = body_obj
    bpy.ops.object.mode_set(mode="OBJECT")
    dg = bpy.context.evaluated_depsgraph_get()
    evaled = body_obj.evaluated_get(dg)
    # HumGen vertices in world space
    vertices = np.asarray([ (evaled.matrix_world @ v.co) for v in evaled.data.vertices ])

    # BFM landmark coordinates in world space
    head_landmarks = np.asarray([ (head_obj.matrix_world @ v.co) for v in (head_obj.data.vertices[i] for i in landmark_indices) ])

    # Looks for closest vertices to the BFM landmarks. The indices thereof are the landmark indices.
    distances = np.linalg.norm(vertices[None,:,:] - head_landmarks[:,None,:], axis=-1)
    new_landmark_indices = np.argmin(distances, axis=1)

    # Debug
    for v in body_obj.data.vertices:
        v.select = False
    for v in (body_obj.data.vertices[i] for i in new_landmark_indices):
        v.select = True

    np.savez(join(DSTDIR,'landmark_indices.npz'), indices = np.asarray(new_landmark_indices,dtype=np.int64))

    #debug_create_face_vertex_object(vertices)

def debug_create_face_vertex_object(vertices):
    m = bpy.data.meshes.new('TEST mesh')
    m.from_pydata(vertices, [], [])
    cpy = bpy.data.objects.new(name = 'TEST', object_data=m)
    bpy.context.scene.collection.objects.link(cpy)
    bpy.context.view_layer.update()