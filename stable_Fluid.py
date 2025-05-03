import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  
#ti.init(arch=ti.opengl)


NDIM = 2 

N = ti.Vector([1024, 1024])

#grid = ti.Vector.field(NDIM, dtype = ti.i32, shape = (N[0], Np[1]))

divergence = ti.field(dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))

P = ti.field(dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))

#origin
O = ti.Vector.field(NDIM, dtype = ti.f32, shape=(1, ))
#length of grid
L = ti.Vector.field(NDIM, dtype = ti.f32, shape=(1, ))
#length of cell 
D = ti.Vector.field(NDIM, dtype = ti.f32, shape=(1,))

#velocity  field
U0 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))
U1 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))

#density field
D0 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))
D1 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))

#source field
S0 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (1, ))
S1 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (1, ))

#omega
W = ti.field(dtype = ti.f32, shape = (N[0] + 2, N[1] + 2))


visc = 0.001
grav2D = ti.Vector([0.0 , -9.8])
grav3D = ti.Vector([0.0 , -9.8, 0.0])

#diffusion const 
kd = 0.1

#source const 
aS = 0.001

dt = 1e-3

@ti.kernel
def initialize_2D(): 
    #Init Grid         
    O[0] = ti.Vector([0.0, 0.0])
    L[0] = ti.Vector([1.0, 1.0]) 
    D[0] = L[0] / N[0] 
    
    #Init Density, Velocity
    for i,j in ti.ndrange((1, N[0] + 1), (1,N[1] + 1)):
        D0[i,j] = ti.Vector([0, 0])
        D1[i,j] = ti.Vector([0, 0])
        U0[i,j] = ti.Vector([0, 0])
        U1[i,j] = ti.Vector([0, 0])  
        W[i ,j] = 0.0  
def SwapU():
    global U0, U1
    U0, U1 = U1, U0

def SwapS():
    global S0, S1
    S0, S1 = S1, S0    

def SwapD():
    global D0, D1
    D0, D1 = D1, D0 

@ti.func
def ForceTerm_D():
    h = 1.0 / N[0]
    
    for i,j in ti.ndrange( (1, N[0] + 1),(1, N[1] + 1)):
        Dy = ( D0[i, j + 1] - D0[i, j - 1]) / (2 * h)
        Dx = ( D0[i + 1, j] - D0[i - 1, j]) / (2 * h)
        D1[i,j] += Dy - Dx
        
@ti.func
def ForceTerm_U():
    h = 1.0 / N[0]
    
    for i,j in ti.ndrange( (1, N[0] + 1),(1, N[1] + 1)):
        Uy = ( U0[i ,j + 1] - U0[i , j -1]) / (2 * h)
        Ux = ( U0[i + 1,j] - U0[i - 1, j]) / (2 * h)
        U1[i,j] += Uy - Ux 
   
@ti.func
def compute_vorticity():
    h = 1.0 / N[0]
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
        dv_dx = (U0[i + 1, j].y - U0[i - 1, j].y) / (2 * h)   
        du_dy = (U0[i, j + 1].x - U0[i, j - 1].x) / (2 * h)  
        W[i, j] = dv_dx - du_dy   
        
        
@ti.func
def apply_vorticity_confinement():
    h = 1.0 / N[0]
    epsilon = 0.01  # 소용돌이 보존 강도
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
        
        
        eta = ti.Vector([ 
                         (ti.abs(W[i + 1,j ]) - ti.abs(W[i - 1, j])) / (2 * h), 
                         (ti.abs(W[i,j + 1]) - ti.abs(W[i, j - 1])) / (2 * h)
                        ])
          
        #정규화 #ti.math.normalize()
        norm = eta.norm()
        N = eta
        if norm > 1e-6:   
            N = eta / norm
         
        N_x = N.x
        N_y = N.y
        w = W[i, j]  
        
        N3 = ti.Vector([N.x, N.y, 0.0])
        W3 = ti.Vector([0.0, 0.0, w]) 
        NcW = ti.math.cross(N3, W3)
        F = epsilon * 5 * h * ti.Vector([NcW.x, NcW.y])
        U1[i, j] = U0[i,j] + F
        
@ti.func
def AddSource_D(mouse_pos: ti.types.vector(2, ti.f32)):
   
    index = ti.cast(mouse_pos * N, ti.i32) 
    radius = 30
    minX = max(index.x - radius, 1)
    minY = max(index.y - radius, 1)  
    maxX = min(index.x + radius, N[0])
    maxY = min(index.y + radius, N[0])
    
        
    for i, j in ti.ndrange( (minX, maxX + 1 ), (minY, maxY + 1)):
        
        dist = (ti.cast((i - index.x)**2 + (j - index.y)**2, ti.f32))
        if dist <= radius * radius:   
            D1[i, j] += ti.Vector([0.08, 0.08])

        
@ti.func
def AddSource_U(mouse_pos: ti.types.vector(2, ti.f32)):
   
    index = ti.cast(mouse_pos * N, ti.i32) 
    radius = 20
    
    minX = max(index.x - radius, 1)
    minY = max(index.y - radius, 1)  
    maxX = min(index.x + radius, N[0])
    maxY = min(index.y + radius, N[0])
    
        
    for i, j in ti.ndrange( (minX, maxX + 1 ), (minY, maxY + 1)): 
        dist = (ti.cast((i - index.x)**2 + (j - index.y)**2, ti.f32))
        if dist <= radius * radius:
            U1[i, j] += ti.Vector([0.0, 2.0]) 
        
@ti.func
def set_bnd(b: ti.i32, x: ti.template()):
    
    for i in ti.ndrange(N[0] + 2):
        x[i, 0] = x[i, 1] if b == 2 else -x[i, 1]  # y=0 경계
        x[i, N[1] + 1] = x[i, N[1]] if b == 2 else -x[i, N[1]]  # y=N+1 경계
    
    for j in ti.ndrange(N[1] + 2):
        x[0, j] = x[1, j] if b == 1 else -x[1, j]  # x=0 경계
        x[N[0] + 1, j] = x[N[0], j] if b == 1 else -x[N[0], j]  # x=N+1 경계
    
    
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N[1] + 1] = 0.5 * (x[1, N[1] + 1] + x[0, N[1]])
    x[N[0] + 1, 0] = 0.5 * (x[N[0], 0] + x[N[0] + 1, 1])
    x[N[0] + 1, N[1] + 1] = 0.5 * (x[N[0], N[1] + 1] + x[N[0] + 1, N[1]])

@ti.func
def Project():
    h = 1.0 / N[0]
    
    # divergence
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):  
        divergence[i, j] = 0.5 * h * ((U0[i+1, j].x - U0[i-1, j].x) + (U0[i, j+1].y - U0[i, j-1].y))
        P[i, j] = 0.0
        
    set_bnd(0, divergence)
    set_bnd(0, P)
    
    # Jacobi iteration
    for k in ti.static(range(20)):
        for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
            P[i, j] = (-divergence[i, j] + P[i-1, j] + P[i+1, j] + P[i, j-1] + P[i, j+1]) / 4.0
    set_bnd(0, P)

    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)): 
        U1[i,j] = U0[i,j]
    #
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)): 
        U1[i, j].x += -0.5 * (P[i+1, j] - P[i-1, j]) / h 
        U1[i, j].y += -0.5 * (P[i, j+1] - P[i, j-1]) / h  
        
    set_bnd(1, U1)  # x 방향 속도
    set_bnd(2, U1)  # y 방향 속도
  

@ti.func
def Advect_D():
    dt0 = dt * N[0]
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
        
        #semi-lagrangian
        pos = ti.Vector([i - U0[i, j].x * dt0, j - U0[i, j].y * dt0])
        
        pos.x = ti.min(ti.max(pos.x, 0.5), N[0] + 0.5)
        pos.y = ti.min(ti.max(pos.y, 0.5), N[1] + 0.5)
        
        i0 = ti.cast(pos.x, ti.i32)
        j0 = ti.cast(pos.y, ti.i32)
        i1 = i0 + 1
        j1 = j0 + 1
        
        s1 = pos.x - i0
        s0 = 1 - s1
        t1 = pos.y - j0
        t0 = 1 - t1 
        
        D1[i, j] = s0 * (t0 * D0[i0, j0] + t1 * D0[i0, j1]) + s1 * (t0 * D0[i1, j0] + t1 * D0[i1, j1])
    set_bnd(0, D1)

@ti.func
def Advect_U():
    dt0 = dt * N[0]
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
        pos = ti.Vector([i - U0[i, j].x * dt0, j - U0[i, j].y * dt0])
        pos.x = ti.min(ti.max(pos.x, 0.5), N[0] + 0.5)
        pos.y = ti.min(ti.max(pos.y, 0.5), N[1] + 0.5)
        
        i0 = ti.cast(pos.x, ti.i32)
        j0 = ti.cast(pos.y, ti.i32)
        i1 = i0 + 1
        j1 = j0 + 1
        s1 = pos.x - i0
        s0 = 1 - s1
        t1 = pos.y - j0
        t0 = 1 - t1
        U1[i, j] = s0 * (t0 * U0[i0, j0] + t1 * U0[i0, j1]) + s1 * (t0 * U0[i1, j0] + t1 * U0[i1, j1])
    set_bnd(1, U1)
    set_bnd(2, U1)
 
    
    
@ti.func
def Diffuse_D():
    for k in ti.static(range(20)):
        for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
            D1[i, j] = (D0[i, j] + kd * (D0[i - 1, j] + D0[i + 1, j] + D0[i, j - 1] + D0[i, j + 1])) / (1 + 4 * kd)
    set_bnd(0, D1)

@ti.func
def Diffuse_U():
    for k in ti.static(range(20)):
        for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
            U1[i, j] = (U0[i, j] + visc * (U0[i - 1, j] + U0[i + 1, j] + U0[i, j - 1] + U0[i, j + 1])) / (1 + 4 * visc)
    set_bnd(1, U1)
    set_bnd(2, U1)

@ti.func
def Diffuse_D_bad():
    for i, j in ti.ndrange((1, N[0] + 1), (1, N[1] + 1)):
        D1[i, j] = D0[i , j] + kd * (D0[i - 1, j] + D0[i + 1, j] + D0[i, j - 1] + D0[i, j + 1] - 4 * D0[i , j])
    #set_bnd(1, U1)
    #set_bnd(2, U1)
    
    
@ti.kernel
def dens_step(mouse_pos: ti.types.vector(2, ti.f32), add_force: ti.i32):
    
    if add_force:
        AddSource_D(mouse_pos)
        SwapD()
    
    Diffuse_D()
    SwapD()
    
    Advect_D() 
    SwapD() 
    
# add Curl
@ti.kernel
def vel_step(mouse_pos: ti.types.vector(2, ti.f32), add_force: ti.i32):
    
    if add_force: 
        AddSource_U(mouse_pos)
        SwapU()
    
    Diffuse_U() 
    SwapU() 
    
    Advect_U()
    SwapU() 
    
    compute_vorticity()
    apply_vorticity_confinement()  
    SwapU()
    
    Project()      
    SwapU()
    
# @ti.kernel
# def vel_step(mouse_pos: ti.types.vector(2, ti.f32), add_force: ti.i32):
    
#     if add_force: 
#         AddSource_U(mouse_pos)
#         SwapU()
    
#     Diffuse_U() 
#     SwapU()
#     Project()      
#     SwapU()
    
#     Advect_U()
#     SwapU()
#     Project()  
#     SwapU()
    
    
    
pixels = ti.field(dtype=ti.f32, shape=(N[0] + 2, N[1] + 2, 3))
 
 
        
@ti.kernel
def update_pixels():
    for i, j in ti.ndrange( (1, N[0] + 1), ( 1, N[1] + 1) ):
        density = D0[i ,j][0] 
        pixels[i,j,0] = ti.min(1.0, density)  # R
        pixels[i,j,1] = ti.min(1.0, density * 0.5)  # G
        pixels[i,j,2] = ti.min(1.0, density * 0.2)  # B

  

current_t = 0.0 
  
window = ti.ui.Window("Taichi Stable Fluid", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))  # 검정색 배경
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# 카메라 설정
camera.position(0.0, 0.0, 1.0)  # 카메라를 정육면체 앞에 배치
camera.lookat(0.0, 0.0, 0.0)   # 정육면체 중심을 바라봄
camera.up(0.0, 1.0, 0.0)        # 상단 방향은 y축
 
mouse_pos = ti.Vector([0.5, 0.5])
add_force = False

initialize_2D() 
while window.running:
    add_force = window.is_pressed(ti.ui.LMB)
    if add_force:
        mx, my = window.get_cursor_pos()
        mouse_pos = ti.Vector([mx, my])
        
    dens_step(mouse_pos, int(add_force))
    vel_step(mouse_pos, int(add_force))
    
    update_pixels()
    canvas.set_image(pixels)  # 2D 유체 렌더링
     
    canvas.scene(scene)  # 3D 씬 렌더링
    window.show()
    add_force = False