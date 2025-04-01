
import taichi as ti
ti.init(arch=ti.cuda)

from BoundaryBox import init_boundary, init_cube_edges, box_vpos, cube_edges

# Draw Boundary Box 
init_boundary()
init_cube_edges()

origin = ti.Vector.field(3, dtype = ti.f32, shape = 1) 
origin[0] = [0,0,0]

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0, 2, 14)
camera.lookat(0,1,0)

while window.running:
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
 
    scene.particles(origin, color=(1, 0, 0) , radius =0.1)
    scene.lines(box_vpos, indices=cube_edges, color = (0.8, 0.4, 0.2), width=5.0, vertex_count=8)

    canvas.scene(scene)
    window.show()