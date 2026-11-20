import numpy as np
from plyfile import PlyData, PlyElement
import json

import math
import mcschematic

# 1. Create a new schematic instance
schem = mcschematic.MCSchematic()

# 2. Place blocks at (x, y, z) coordinates
schem.setBlock((0, 0, 0), "minecraft:stone_bricks")
schem.setBlock((0, 1, 0), "minecraft:oak_planks")
schem.setBlock((1, 0, 0), "minecraft:oak_stairs[facing=north,half=bottom]")



#define the size and detail
scale=10

#define the bounds

min_x = -0.3
max_x = 0.3
max_z = 0.3
min_z = -0.3

#because model weird
x = 'y'
y = 'z'
z = 'x'

#for models that are well writeen
x = 'x'
y = 'y'
z = 'z'


import json


color_minecraft = {}

with open("block_colors.json", "r") as file:
    data = json.load(file)

print(type(data))

for i in data.keys():
     #print(data[i]["r"])
     #if "wool" in i:
     color_minecraft[(data[i]["r"],data[i]["g"],data[i]["b"])] = i

print(color_minecraft)


blocks_list = {}
density = 0


with open('scene.ply', 'rb') as f:

     plydata = PlyData.read(f)
     splats = plydata['vertex']
     number_splats = 356453
     #print(splats.properties)

     #we create the list of colors rgb
     SH_C0 = 0.28209479177387814
     f_dc = np.stack([splats['f_dc_0'], splats['f_dc_1'], splats['f_dc_2']], axis=1)
     rgb = np.clip(0.5 + SH_C0 * f_dc, 0, 1)   # floats 0–1; multiply by 255 for ints

     for i in range (number_splats):


          splat = splats[i]
          color = tuple(int(c * 255) for c in rgb[i])
           
          #if ((splat[x] > min_x and splat[x] < max_x) and (splat[z] > min_z and splat[z] < max_z)) :

          block = "minecraft:dripstone_block"
          mini = 1000000
          for mine in color_minecraft.keys():
               distance = math.sqrt((mine[0]-color[0])**2+(mine[1]-color[1])**2+(mine[2]-color[2])**2)
               if distance < mini:
                    block = color_minecraft[mine]
                    mini = distance

          if (color in color_minecraft):
               block = color_minecraft[color]

          if (int(splat[x]*scale),int(splat[y]*-scale),int(splat[z]*scale), block) not in blocks_list:
               blocks_list[(int(splat[x]*scale),int(splat[y]*-scale),int(splat[z]*scale), block)] = 0

          blocks_list[(int(splat[x]*scale),int(splat[y]*-scale),int(splat[z]*scale), block)] += 1
          print(f"x = {splat[x]}, y= {splat[y]}, z={splat[z]}")


for final in blocks_list.keys():
     if (blocks_list[final] > density):
          print([final[3]])
          schem.setBlock((final[0],final[1],final[2]),final[3])


# 3. Save the file (outputs "my_build.schem" in the current folder)
schem.save(".", "my_build", mcschematic.Version.JE_1_20_1)