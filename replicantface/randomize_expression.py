from pathlib import Path
import random
import numpy as np

import bpy
from HumGen3D import Human, HumGenException


EXPRESSION_NAMES = [
'shapekeys/expressions/Expressive/constipated.npz',
'shapekeys/expressions/Expressive/angry_yell.npz',
'shapekeys/expressions/Expressive/flabbergasted.npz',
'shapekeys/expressions/Expressive/unpleasant_surprise.npz',
'shapekeys/expressions/Expressive/slightly_disgusted.npz',
'shapekeys/expressions/Neutral/kiss.npz',
'shapekeys/expressions/Neutral/intrigued.npz',
'shapekeys/expressions/Neutral/slight_surprise.npz',
'shapekeys/expressions/Neutral/kiss_closed_eyes.npz',
'shapekeys/expressions/Neutral/concentrated.npz',
'shapekeys/expressions/Neutral/talking.npz',
'shapekeys/expressions/Base Shapekeys/Disgust.npz',
'shapekeys/expressions/Base Shapekeys/Brow Raise Right.npz',
'shapekeys/expressions/Base Shapekeys/Smile.npz',
'shapekeys/expressions/Base Shapekeys/Mouth Turn Left.npz',
'shapekeys/expressions/Base Shapekeys/Lip Press.npz',
'shapekeys/expressions/Base Shapekeys/Happy.npz',
'shapekeys/expressions/Base Shapekeys/Frown.npz',
'shapekeys/expressions/Base Shapekeys/Pucker.npz',
'shapekeys/expressions/Base Shapekeys/Brow Raise Left.npz',
'shapekeys/expressions/Base Shapekeys/Mouth Turn Right.npz',
'shapekeys/expressions/Base Shapekeys/Surprise.npz',
'shapekeys/expressions/Base Shapekeys/Angry.npz',
'shapekeys/expressions/Base Shapekeys/Cheeck Suck.npz',
'shapekeys/expressions/Base Shapekeys/Sad.npz',
'shapekeys/expressions/Base Shapekeys/Lip Funnel.npz',
'shapekeys/expressions/Happy/cheeky_smile.npz',
'shapekeys/expressions/Happy/surprised_smile.npz',
'shapekeys/expressions/Happy/happy_smile.npz',
'shapekeys/expressions/Happy/big_surprise.npz',
'shapekeys/expressions/Sad/sad_smile.npz',
]

MOUTH_OPENED = [
'shapekeys/expressions/Expressive/angry_yell.npz',
'shapekeys/expressions/Happy/big_surprise.npz',
'shapekeys/expressions/Expressive/flabbergasted.npz',
'shapekeys/expressions/Expressive/unpleasant_surprise.npz',
'shapekeys/expressions/Base Shapekeys/Surprise.npz',
'shapekeys/expressions/Base Shapekeys/Lip Funnel.npz',
'shapekeys/expressions/Happy/big_surprise.npz',
]

BLINK_LEFT = 'shapekeys/expressions/Base Shapekeys/Blink Left.npz'
BLINK_RIGHT = 'shapekeys/expressions/Base Shapekeys/Blink Right.npz'


class ExpressionController:
    def __init__(self, hum : Human):
        self.keys_by_name = { k.name:k for k in hum.expression.keys }
        self.hum = hum

    def clear(self):
        for k in self.keys_by_name.values():
            k.value = 0.
    
    def apply_weights(self, expressions : dict[Path, float]):
        values_by_name = { str(Path(path).stem):value for path,value in expressions.items() }
        need_update = False
        for name, path in zip(values_by_name.keys(), expressions.keys()):
            if not name in self.keys_by_name:
                # Load as new key. Warning: might create duplicates.
                # Also warning: set's all other weights to 0
                self.hum.expression.set(path)
                need_update = True
        if need_update:
            self.keys_by_name = { k.name:k for k in self.hum.expression.keys }
        for name, shapekey in self.keys_by_name.items():
            try:
                shapekey.value = values_by_name[name]
                print ("Expression set key: ", name, values_by_name[name])
            except KeyError:
                shapekey.value = 0.
                print ("Expression key error on ", name, ". zeroing")


def randomize_expression(hum : Human, p_neutral : float = 0.2, p_eyes_closed : float = 0.4, p_open_mouth : float = 0.4):
    ctrl = ExpressionController(hum)
    weights = {}
    if random.uniform(0.,1.) < p_neutral:
        # Neutral
        if random.uniform(0., 1.) < p_open_mouth:
            a, b = MOUTH_OPENED[:2] # Both combined give a relatively neutral expression
            amount = random.uniform(0.5, 1.)
            weights[a] = 0.5*amount
            weights[b] = 0.5*amount
    else:
        num_expression = random.choice([1, 2, 3])
        names = np.random.choice(EXPRESSION_NAMES, size=(num_expression,), replace=False)
        has_mouth_open =  any((x in MOUTH_OPENED) for x in names)
        if not has_mouth_open and random.uniform(0.,1.) < p_open_mouth:
            names[0] = random.choice(MOUTH_OPENED)
        for name in names:
            # With ca. 50% probability the sample is in [0.8,1.]
            weights[name] = random.betavariate(4.,1.2)
        # ("Expressions: ", weights)

    # Close eyes with some probability
    if random.uniform(0., 1.) < p_eyes_closed:
        w = random.choice([1.,0.5])
        weights[BLINK_LEFT] = w
        weights[BLINK_RIGHT] = w

    ctrl.apply_weights(weights)


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
    randomize_expression(hum)