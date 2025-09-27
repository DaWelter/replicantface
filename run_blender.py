#!/bin/env python

from typing import Literal
from os.path import dirname, join
import asyncio
import sys

blender="<blender executable>"
scene=join(dirname(__file__),'human_scene-v3.blend')
script=join(dirname(__file__),'make_faces.py')
outdir="<output directory>"
device : Literal['CPU','CUDA'] = 'CUDA'

start=0
count=5000


async def output_filter(input_stream : asyncio.StreamReader, output_stream):
    while not input_stream.at_eof():
        output = await input_stream.readline()
        if not output.startswith(b"Error: Vertex"): # There is a lot of spam with that error. Doesn't seem to be harmful.
            output_stream.buffer.write(output)


async def run_blender(i):
    # Based on https://stackoverflow.com/questions/36277995/how-to-run-python-subprocess-and-stream-but-also-filter-stdout-and-stderr
    process = await asyncio.create_subprocess_exec(blender, *[
        scene,
        '-o', join(outdir,f'face_{i:05}.jpg'),
        *'-b --offline-mode'.split(' '), # --log 0
        '-P', script, '--',
        '--cycles-device', device,
    ], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    await asyncio.gather(
        output_filter(process.stderr, sys.stderr),
        output_filter(process.stdout, sys.stdout),
    )

    sys.stderr.flush()
    sys.stdout.flush()

    await process.communicate()

end=start+count
for i in range(start, end):
    asyncio.run(run_blender(i))
