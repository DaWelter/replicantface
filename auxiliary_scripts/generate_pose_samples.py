"""Entry point for generating pose samples for debugging.

Generate random position for face and camera like in the real randomization script
and write out the modelview matrices.
"""
import bpy
from pathlib import Path
from HumGen3D import Human
import sys
import sys
import numpy as np

if __name__ == '__main__':
    # Trigger reimport when the script is run again.
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]


from replicantface import (
                           find_hum, 
                           sample_pose,
                           compute_model_view_matrix)


if __name__ == '__main__':
    scene = bpy.data.scenes['Scene']
    bpy.context.window.scene = scene
    
    cam = scene.objects['Camera']
    hum = find_hum()
    hum_obj = hum.objects.rig

    NUM_SAMPLES = 10000

    outputs = []

    for i in range(NUM_SAMPLES):
        sample_pose(wide_distribution=True).apply_to_scene(cam, hum)
        m = compute_model_view_matrix(hum_obj, cam)
        outputs.append(m)
        # This is very very slow, but needed to update the matrices.
        bpy.context.view_layer.update()
    
    # shape (n,3,3)
    np.savez_compressed('/tmp/head-poses.npz', modelview = outputs)