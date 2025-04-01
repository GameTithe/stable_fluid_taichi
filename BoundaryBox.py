
import taichi as ti


box_vpos = ti.Vector.field(3, dtype=ti.f32, shape = 8)

box_width= 3
box_height = 2
def init_boundary():
    #top
    box_vpos[0] = [box_width, box_height, box_width]
    box_vpos[1] = [-box_width, box_height, box_width]
    box_vpos[2] = [box_width, box_height, -box_width]
    box_vpos[3] = [-box_width, box_height, -box_width]
    
    #bottom
    box_vpos[4] = [box_width, 0, box_width]
    box_vpos[5] = [-box_width, 0, box_width]
    box_vpos[6] = [box_width, 0, -box_width]
    box_vpos[7] = [-box_width, 0, -box_width]

cube_edges = ti.field(dtype=ti.i32, shape=24)
edges = [
    # top face
    0, 1, 1, 3, 3, 2, 2, 0,
    # bottom face
    4, 5, 5, 7, 7, 6, 6, 4,
    # vertical edges
    0, 4, 1, 5, 2, 6, 3, 7
]

def init_cube_edges():
    for i in range(24):
        cube_edges[i] = edges[i]

