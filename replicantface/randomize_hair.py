import bpy
from math import pi
from os.path import join, dirname
from pathlib import Path
from HumGen3D import Human, HumGenException

import random

if __name__ == '__main__':
    import sys
    sys.path.append(Path(bpy.data.filepath).parent.as_posix())
    for k in list(sys.modules.keys()):
        if 'replicantface' in k:
            del sys.modules[k]


from replicantface.utils import gaussian, find_hum, HeadCoverage


class Hair:
    def __init__(self):
        self.SHORT_HAIR_STYLES = [
                    'Buzzcut Fade',
                    'Buzzcut Curly Fade',
                    'Short Curly Fade',
                ]

        def get_hairstyles(gender):
            root = Path(bpy.app.binary_path).parent / 'humgen-assets'
            hair_folder = root / 'hair' / 'head' / gender
            filenames = [ p.relative_to(root) for p in hair_folder.rglob('*.json') ]
            output = { str(fn.stem):fn for fn in filenames }
            assert all(x in output for x in self.SHORT_HAIR_STYLES)
            return output


        self.HAIRSTYLES = {
            'male' : get_hairstyles('male'),
            'female' : get_hairstyles('female')
        }

    def _compute_hair_params(self, age : int, kinda_old_threshold : float, really_old_age : float):
        # Young age:
        #       Lightness, Redness
        # 0 -   0.0      , 0. ,
        # 0 -   0.25     , 0.25
        # 0 -   1.5      , 0.5,    
        # 0 -   3.0      , 0.75,   
        # 0 -   4.0      , 1.0

        # L   /|
        # |  /||
        # | /||allowed
        # |/||||
        # --------> R

        # Little bit old
        # Lightness   Redness
        # 0.5 -1      0.
        # 0.5 - 1     0.25
        # 1.5 - 4     0.5
        # L   | 0.3 - 0.6
        # |  || 
        # | ||  0 - 0.25
        # | 
        # -----------> R

        # Very aged:
        # Lightness   Redness
        # 4,          0

        normed_age = max(0., min(1., (age-kinda_old_threshold) / (really_old_age-kinda_old_threshold)))
        min_lightness = max(0., min(1., random.normalvariate(normed_age, 0.2)))*4.
        max_lightness = 4.
        # Young
        lightness = random.uniform(min_lightness, max_lightness)
        young_min_redness = lightness / 4.
        young_max_redness = 1.
        # Really old
        old_min_redness = 0.
        old_max_redness = 1. - lightness / 4.
        # Current age
        min_redness = young_min_redness * (1.-normed_age) + normed_age * old_min_redness
        max_redness = young_max_redness * (1.-normed_age) + normed_age * old_max_redness
        redness = random.uniform(min_redness, max_redness)

        snp = gaussian(normed_age, 0.5, 0.2) * 0.3

        return lightness, redness, snp

    def randomize(self, hum : Human, age : int, head_cover : HeadCoverage):
        # CAUTION: once set, it can't be deleted :-(

        base_lightness, base_redness, base_snp = self._compute_hair_params(age, 30., 70.)

        if 1: # brows
            preset_eyebrows : list[bpy.types.ParticleSystem] = [
                    mod.particle_system
                    for mod in hum.objects.body.modifiers
                    if hasattr(mod, 'particle_system') and mod.particle_system.name.lower().startswith('eyebrows') ]
            brows : bpy.types.ParticleSystem = random.choice(preset_eyebrows)

            radius = random.uniform(0.005, 0.02)
            brows.settings.root_radius = radius
            brows.settings.tip_radius = radius / 5.
            hum.hair.eyebrows.set(brows.name)

            lightness, _, _ = self._compute_hair_params(age, 50., 90.)
            hum.hair.eyebrows.lightness.value = lightness
            hum.hair.eyebrows.redness.value = base_redness
        

        hue = random.normalvariate(0.5, 0.02) #random.uniform(0.46, 0.51)
        roughness = random.uniform(0.1, 0.5)

        if head_cover.top_covered != 'tight':
            if head_cover.top_covered == 'loose':
                selected = self.HAIRSTYLES[hum.gender][random.choice(self.SHORT_HAIR_STYLES)]
            else:
                selected : Path = random.choice(list(self.HAIRSTYLES[hum.gender].values()))

            hum.hair.regular_hair.set(str(selected), context = bpy.context)
            hum.hair.update_hair_shader_type('accurate')

            hum.hair.regular_hair.lightness.value = base_lightness
            hum.hair.regular_hair.redness.value = base_redness
            hum.hair.regular_hair.salt_and_pepper.value = base_snp

            # Don't work. Hack around it ...
            #hum.hair.regular_hair.hue = hue
            #hum.hair.regular_hair.roughness = roughness
            m = bpy.data.materials['.HG_Hair_Head'].node_tree.nodes['HG_Hair']
            m.inputs["Hue"].default_value = hue
            m.inputs["Roughness"].default_value = roughness
            #bpy.data.node_groups["HG_Hair_V4"].nodes["Principled Hair BSDF"].inputs['Coat'].default_value = 0.513636
        
        if hum.gender == 'male' and head_cover.allow_beard:
            lightness, _, snp = self._compute_hair_params(age, 20., 60)
            hum.hair.face_hair.set_random(context = bpy.context)
            hum.hair.face_hair.lightness.value = lightness
            hum.hair.face_hair.redness.value = base_redness
            m = bpy.data.materials['.HG_Hair_Face'].node_tree.nodes['HG_Hair']
            m.inputs["Hue"].default_value = hue
            m.inputs["Roughness"].default_value = roughness
            hum.hair.face_hair.salt_and_pepper.value = snp


if __name__ == '__main__':
    hum = find_hum()
    age = random.randint(20,70)
    hum.age.set(age)

    hair = Hair()
    hair.randomize(hum, age, HeadCoverage('none', False))