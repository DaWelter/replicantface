# Replicant Face

This project is about generating synthetic faces with pose annotations.
It's based on [Blender](https://www.blender.org/) and the [Human Generator](https://blendermarket.com/products/humgen3d) addon.

Only the code for randomizing the scene and exporting the data is in this repo.
Due to licensing reasons the assets can't be published. And therefore the scene
file is also not included.

## Setup

TODO: scene screenshot, how to reproduce the scene and head data.

## Usage

### Batchprocessing example
```Python
#!/bin/env python

# Run blender without gui. Only a certain batchsize of images then restart
# until desired number of images was rendered.
from typing import Literal
import subprocess
from os.path import dirname, join

blender="<install dir>/blender"
scene=join(dirname(__file__),'human_scene.blend')
script=join(dirname(__file__),'make_faces.py')
outdir="<destination dir>"
batchsize=100
device : Literal['CPU','CUDA'] = 'CUDA'

count=200
start=0

end=start+count
for i in range(start, end, batchsize):
    my_batchsize = min(batchsize, end-i)
    subprocess.check_call([
        blender,
        scene,
        *'--log 0 -b --offline-mode'.split(' '),
        '-P', script, '--',
        '--cycles-device', device,
        '--start', str(i), '--count', str(my_batchsize), '-o', outdir
    ])
```

### Postprocessing example

TODO

## License

MIT?