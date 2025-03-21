import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

NDIM = 2 

N = ti.Vector([128, 128])
#grid = ti.Vector.field(NDIM, dtype = ti.i32, shape = (N[0], N[1]))
divergence = ti.field(dtype = ti.f32, shape = (N[0], N[1]))

P = ti.field(dtype = ti.f32, shape = (N[0], N[1]))
O = ti.Vector.field(NDIM, dtype = ti.f32, shape=(1, ))
L = ti.Vector.field(NDIM, dtype = ti.f32, shape=(1, ))
D = ti.Vector.field(NDIM, dtype = ti.f32, shape=(1,))

#velocity  field
U0 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0], N[1]))
U1 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0], N[1]))

#density field
D0 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0], N[1]))
D1 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (N[0], N[1]))


#source field
S0 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (1, ))
S1 = ti.Vector.field(NDIM, dtype = ti.f32, shape = (1, ))

visc = 0.0001
grav2D = ti.Vector([0.0 , -9.8])
grav3D = ti.Vector([0.0 , -9.8, 0.0])

#diffusion const 
kd = 0.5

#source const 
aS = 0.5

dt = 1e-1

@ti.kernel
def initialize_2D(): 
    # for i, j in ti.ndrange(N[0], N[1]):
    #     grid[i, j] = ti.Vector([i, j]) 
               
    O[0] = ti.Vector([0.0, 0.0])
    L[0] = ti.Vector([1.0, 1.0])
    D[0] = L[0] / N[0] 
     
     
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
def AddSource():
    for i, j in ti.ndrange(N[0], N[1]):
        D1[i,j] += S0[0] * dt
 
@ti.func
def Project():
    h  = 1.0 / float(N[0])
    
    for i, j in ti.ndrange(1, int(N[0]) - 1, 1 , int( N[1]) - 1) :  
        divergence[i, j] = -0.5 * h * ((U1[i+1, j].x - U1[i-1, j].x) + (U1[i, j+1].y - U1[i, j-1].y))

        P[i , j] = 0
        
    # boundary
    
    for k in ti.static(20):
        for i, j in ti.ndrange(1 , int(N[0]) - 1, 1, int(N[1]) - 1 ):
            P[i, j] = (divergence[i, j] + P[i-1, j] + P[i+1, j] + P[i, j-1] + P[i, j+1]) / 4.0

    for i, j in ti.ndrange(1, int(N[0]) - 1, 1, int(N[1]) - 1):
        # x방향 속도 보정: 중앙 차분으로 압력의 기울기를 계산
        U0[i, j].x -= 0.5 * (P[i+1, j] - P[i-1, j]) / h
        # y방향 속도 보정: 중앙 차분으로 압력의 기울기를 계산
        U0[i, j].y -= 0.5 * (P[i, j+1] - P[i, j-1]) / h 
      
@ti.func
def Advect():
    for i, j in ti.ndrange(N[0], N[1]):
        pos = ti.Vector([i - U0[i,j].x * dt, j - U0[i,j].y * dt])
        #boundary check
        i0 = ti.cast(i, ti.i32)
        j0 = ti.cast(j, ti.i32)
        i1 = i0 + 1
        j1 = j0 + 1

        #interpolation
        s1 = pos.x - i0
        s0 = 1 - s1
        t1 = pos.y - j0
        t0 = 1 - t1
          
        D1[i, j] = s0 * (t0 * D0[i0, j0] + t1 * D0[i0, j1]) + s1 * (t0 * D0[i1, j0] + t1 * D0[i1, j1])


@ti.func
def Diffuse():

    # boundary check
    
    for k in ti.static(range(20)) :
        for i, j in ti.ndrange(N[0], N[1]):
            D1[i, j] = (D0[i ,j] + kd * ( D1[i - 1, j] + D1[i + 1, j] + D1[i,j -1] + D1[i, j  + 1])) / (1 + 4 * kd)
 
    # boundary check
    
@ti.kernel
def dens_step():
    AddSource()
    SwapD()
    
    Diffuse()
    SwapD()
    
    Advect()
    
@ti.kernel
def vel_step():
    
    AddSource()
    SwapU()
    
    Project()
    SwapU()
    
    Advect()
    Project() 

    
def step(): 
    dens_step()
    vel_step()
    
    
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0 

initialize_2D()
while window.running:
    vel_step()
    dens_step()


    step() 