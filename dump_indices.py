import bpy
import numpy as np
from os.path import join, dirname

DSTDIR = dirname(bpy.data.filepath)

if 1:
    # Dumps indices of selected vertices.
    # Usage: 
    #   Select vertices from the head.
    #   Then run this script.
    #   The face rendering script will export those vertices into the annotations file.
    #   Use the data in postprocessing to compute bounding box and extract landmarks.
    c = bpy.context.active_object
    indices = []
    for i,v in enumerate(c.data.vertices):
        if v.select:
            indices.append(i)
    print('selected num verts: ', len(indices))
    np.savez(join(DSTDIR,'head_indices.npz'), indices = np.asarray(indices,dtype=np.int64))

def debug_create_face_vertex_object(vertices):
    m = bpy.data.meshes.new('TEST mesh')
    m.from_pydata(vertices, [], [])
    cpy = bpy.data.objects.new(name = 'TEST', object_data=m)
    bpy.context.scene.collection.objects.link(cpy)
    bpy.context.view_layer.update()

if 0:
    # Extracts landmark indices based on reference model.
    # Usage:
    #   - Make scene with a Hum Gen human.
    #   - Overlay a reference head model so it roughly matches the Hum Gen face.
    #   For example the BFM head model which was used here.
    #   - Adjust the `landmark_indices` variable to contain the landmark vertices
    #   in the reference model.
    #   - Run this script. It should find the closest vertices in the Hum Gen model
    #   and write their indices to `landmark_indices.npz`.
    #   Then they can be used in postprocessing to grab the corresponding points
    #   from the vertex annotations. Thus you can get 3d landmark annotations.
    hum_obj = bpy.context.scene.objects['HG_Clay']
    head_obj = bpy.context.scene.objects['face']
    headbone = hum_obj.pose.bones['head']
    headbone_model_world = hum_obj.matrix_world @ headbone.matrix
    translation = (headbone_model_world.inverted() @ head_obj.matrix_world).col[3]
    print(translation)

    landmark_indices = np.array([17440, 17716, 17222, 16770, 33981, 34802, 35355, 35777, 36091,
        36403, 36806, 37333, 38086, 29006, 28546, 28036, 28276, 29704,
        30401, 30791, 30996, 31163, 31902, 32068, 32276, 32670, 33376,
            8161,  8177,  8187,  8192,  6515,  7243,  8204,  9163,  9883,
            1959,  3887,  5048,  6216,  4674,  3513,  9956, 11223, 12384,
        14327, 12656, 11495,  5522,  6025,  7495,  8215,  8935, 10395,
        10795,  9555,  8836,  8236,  7636,  6915,  5909,  7384,  8223,
            9064, 10537,  8829,  8229,  7629])

    for v in head_obj.data.vertices:
        v.select = False
    for v in (head_obj.data.vertices[i] for i in landmark_indices):
        v.select = True

    # The following works only in object mode. Else the eval result won't have vertices. Dafuq?
    body_obj = bpy.context.scene.objects['HG_Body']
    bpy.context.view_layer.objects.active = body_obj
    bpy.ops.object.mode_set(mode="OBJECT")
    dg = bpy.context.evaluated_depsgraph_get()
    evaled = body_obj.evaluated_get(dg)
    vertices = np.asarray([ (evaled.matrix_world @ v.co) for v in evaled.data.vertices ])

    head_landmarks = np.asarray([ (head_obj.matrix_world @ v.co) for v in (head_obj.data.vertices[i] for i in landmark_indices) ])

    distances = np.linalg.norm(vertices[None,:,:] - head_landmarks[:,None,:], axis=-1)
    body_indices = np.argmin(distances, axis=1)

    for v in body_obj.data.vertices:
        v.select = False
    for v in (body_obj.data.vertices[i] for i in body_indices):
        v.select = True

    np.savez(join(DSTDIR,'landmark_indices.npz'), indices = np.asarray(body_indices,dtype=np.int64))

    #debug_create_face_vertex_object(vertices)