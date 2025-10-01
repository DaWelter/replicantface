#!/bin/env python

from typing import Literal
from os.path import dirname, join
import asyncio
import sys

blender="<blender executable>"
scene=join(dirname(__file__),'human_scene-v3.blend')
script=join(dirname(__file__),'make_faces.py')
device : Literal['CPU','CUDA'] = 'CUDA'


async def output_filter(input_stream : asyncio.StreamReader, output_stream):
    while not input_stream.at_eof():
        output = await input_stream.readline()
        if not output.startswith(b"Error: Vertex"): # There is a lot of spam with that error. Doesn't seem to be harmful.
            output_stream.buffer.write(output)


async def run_blender(scene : str, output_name : str, *additional_args : str):
    # Based on https://stackoverflow.com/questions/36277995/how-to-run-python-subprocess-and-stream-but-also-filter-stdout-and-stderr
    cmd = [
        scene,
        '-o', output_name,
        *'-b --offline-mode'.split(' '), # --log 0
        '-P', script, '--',
        '--cycles-device', device,
    ]
    process = await asyncio.create_subprocess_exec(blender, *cmd, *additional_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    await asyncio.gather(
        output_filter(process.stderr, sys.stderr),
        output_filter(process.stdout, sys.stdout),
    )

    sys.stderr.flush()
    sys.stdout.flush()

    await process.communicate()


if 0:
    outdir="<out dir>"
    # Default fully randomized faces
    start=0
    count=100
    end=start+count
    for i in range(start, end):
        asyncio.run(run_blender(scene,join(outdir,f'face_{i:05}.jpg')))
else:
    # Random posed face, kept fix for a number of iterations, while environment and expressions vary.
    num_variations = 32
    outdir="<out dir>"
    start=0
    count=64
    end=start+count
    for i in range(start, end):
        asyncio.run(run_blender(scene,join(outdir,f'face_{i:05}_00.jpg'), '--dump-new-human', '/tmp/posed_human.blend'))
        for j in range(1,num_variations+1):
           asyncio.run(run_blender('/tmp/posed_human.blend',join(outdir,f'face_{i:05}_{j:02}.jpg'), '--randomize-existing'))