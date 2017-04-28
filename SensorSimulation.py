from numpy import sin,pi
from numpy.random import randn
from numpy import array,sin
from pandas import DataFrame
from noise import pnoise1
# =================================================================================
# Each class provides
#      step()            - take a step of size dt
#      sense()           - current value + measurement error
#      step_and_sense()  - combination of step and sense
# =================================================================================
def smooth_noise( dx=0.0005, std=1. ):
    d =  0.
    dt = dx
    s = std
    def next():
        nonlocal d, dt, s
        d += dt
        return s*pnoise1(d,repeat=1000000)
    return next
# TODO replace [va]_std with upd = smooth_noise()
#              use with upd()  instead of randn()*self.[vz]_std.... 
# =================================================================================
def measurements( df, sel ):
    m = df.loc[:,sel].as_matrix()
    return m.reshape( m.shape[0], len(sel), 1)
# =================================================================================
class Const1DVelocitySensor(object):
    ''' Simulate an object moving at constant velocity'''
    def __init__(self, x=0, v=1, dt=0.1, noise_std=1.):
        self.x         = float(x)
        self.v         = float(v)
        self.noise_std = float(noise_std)

        self.dt        = float(dt)
        self.dx        = float(v*dt)

    def step(self):
        self.x += self.dx
        return self.x

    def sense(self):
        return self.x + randn() * self.noise_std

    def step_and_sense(self):
        self.x += self.dx
        return ( self.x + randn() * self.noise_std, self.x )

    def step_and_track(self, n):
        x,t_x = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'x.true' : t_x } )

# ---------------------------------------------------------------------------------
class Const2DVelocitySensor(object):
    ''' Simulate an object moving at constant velocity'''
    def __init__(self, x=(0,0), v=(1,0), dt=0.1, noise_std=1.):
        self.x_sensor = Const1DVelocitySensor( x=x[0], v=v[0], dt=dt, noise_std=noise_std )
        self.y_sensor = Const1DVelocitySensor( x=x[1], v=v[1], dt=dt, noise_std=noise_std )

    def step(self):
        x = self.x_sensor.step()
        y = self.y_sensor.step()
        return (x,y)

    def sense(self):
        x = self.x_sensor.sense()
        y = self.y_sensor.sense()
        return (x,y)

    def step_and_sense(self):
        x,t_x = self.x_sensor.step_and_sense()
        y,t_y = self.y_sensor.step_and_sense()
        return (x,y,t_x,t_y)

    def step_and_track(self, n):
        x,y,t_x,t_y = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'y' : y, 'x.true' : t_x, 'y.true' : t_y } )
# =================================================================================
class PseudoConst1DVelocitySensor(object):
    ''' Simulate an object moving at almost constant velocity
        TODO: velocity is currently a random walk - replace with smooth noise....
    '''
    def __init__(self, x=0, v=1, dt=0.1, v_std=0., noise_std=1.):
        self.x         = float(x)
        self.v         = float(v)
        self.noise_std = float(noise_std)

        self.dt        = float(dt)
        self.v_std     = float(v_std)

    def step(self):
        self.v += randn() * self.v_std;
        self.x += self.v * self.dt
        return ( self.x, self.v )

    def sense(self):
        return (self.x + randn() * self.noise_std, self.v)

    def step_and_sense(self):
        self.v += randn() * self.v_std;
        self.x += self.v * self.dt

        return (self.x + randn() * self.noise_std, self.x, self.v)

    def step_and_track(self, n):
        x,t_x,xDot = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'xDot' : xDot, 'x.true' : t_x } )
# ---------------------------------------------------------------------------------
class PseudoConst2DVelocitySensor(object):
    ''' Simulate an object moving at constant velocity - TODO smooth_noise'''
    def __init__(self, x=(0,0), v=(1,0), dt=0.1, v_std=0., noise_std=1.):
        self.x_sensor = PseudoConst1DVelocitySensor( x=x[0], v=v[0], dt=dt, v_std=v_std, noise_std=noise_std )
        self.y_sensor = PseudoConst1DVelocitySensor( x=x[1], v=v[1], dt=dt, v_std=v_std, noise_std=noise_std )

    def step(self):
        x,xv = self.x_sensor.step()
        y,yv = self.y_sensor.step()
        return (x,y,xv,yv)

    def sense(self):
        x,xv = self.x_sensor.sense()
        y,yv = self.y_sensor.sense()
        return (x,y,xv,yv)

    def step_and_sense(self):
        x,t_x,xDot = self.x_sensor.step_and_sense()
        y,t_y,yDot = self.y_sensor.step_and_sense()
        return ( x,y, xDot, yDot, t_x, t_y )

    def step_and_track(self, n):
        x,y, xDot, yDot, t_x, t_y = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'xDot' : xDot, 'x.true' : t_x,
                            'y' : y, 'yDot' : yDot, 'y.true' : t_y } )
# =================================================================================
class Const1DAccelerationSensor(object):
    ''' Simulate an object moving at constant acceleration'''
    def __init__(self, x=0, v=0, a=0, dt=0.1, x_noise_std=0., v_noise_std=0. ):
        self.x           = float(x)
        self.v           = float(v)
        self.a           = float(a)

        self.v_noise_std = float(v_noise_std)
        self.x_noise_std = float(x_noise_std)

        self.dt          = float(dt)
        self.dv          = float(a*dt)

    def step(self):
        self.v      += self.dv
        self.x      += self.v * self.dt
        return (self.x, self.v)

    def sense(self):
        return (self.x + randn() * self.x_noise_std,
                self.v + randn() * self.v_noise_std)

    def step_and_sense(self):
        self.v      += self.dv
        self.x      += self.v * self.dt

        return (self.x + randn() * self.x_noise_std,
                self.v + randn() * self.v_noise_std, self.x, self.v )

    def step_and_track(self, n):
        x,xDot,t_x,t_v = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'xDot' : xDot, 'x.true' : t_x, 'xDot.true' : t_v } )
# ---------------------------------------------------------------------------------
class Const2DAccelerationSensor(object):
    ''' Simulate an object moving at constant acceleration'''
    def __init__(self, x=(0,0), v=(1,0), a=(1,0), dt=0.1, x_noise_std=1., v_noise_std=0. ):
        #self.x_sensor = Const1DyAccelerationSensor( x=x[0], v=v[0], a=a[0], dt=dt, v_std, noise_std=noise_std )
        #self.y_sensor = Const1DyAccelerationSensor( x=x[1], v=v[1], a=a[1], dt=dt, v_std, noise_std=noise_std )
        self.x_sensor = Const1DAccelerationSensor( x=x[0], v=v[0], a=a[0], dt=dt, noise_std=noise_std )
        self.y_sensor = Const1DAccelerationSensor( x=x[1], v=v[1], a=a[1], dt=dt, noise_std=noise_std )

    def step(self):
        x,xv = self.x_sensor.step()
        y,yv = self.y_sensor.step()
        return (x,y, xv,yv)

    def sense(self):
        x,xv = self.x_sensor.sense()
        y,yv = self.y_sensor.sense()
        return (x,y, xv,yv)

    def step_and_sense(self):
        x,xDot,t_x,t_xDot = self.x_sensor.step_and_sense()
        y,yDot,t_y,t_yDot = self.y_sensor.step_and_sense()
        return (x, y, xDot, yDot, t_x, t_y, t_xDot, t_yDot)

    def step_and_track(self, n):
        x,xDot,t_x,t_xDot,y,yDot,t_y,t_yDot = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x'         : x,      'y'         : y,
                            'xDot'      : xDot,   'yDot'      : yDot,
                            'x.true'    : t_x,    'y.true'    : t_y,
                            'xDot.true' : t_xDot, 'yDot.true' : t_yDot
                            } )
# =================================================================================
class PseudoConst1DAccelerationSensor(object):
    ''' Simulate an object moving at almost constant acceleration '''
    def __init__(self, x=0, v=0, a=0, dt=0.1, x_noise_std=1., v_noise_std=1., a_std=0. ):
        self.x           = float(x)
        self.v           = float(v)
        self.a           = float(a)
        self.dt          = float(dt)

        self.a_std       = float(a_std)
        self.v_noise_std = float(v_noise_std)
        self.x_noise_std = float(x_noise_std)

    def step(self):
        self.a      += randn() * self.a_std;
        self.v      += self.a*self.dt
        self.x      += self.v*self.dt

        return (self.x, self.v, self.a)

    def sense(self):
        return (self.x + randn() * self.x_noise_std,
                self.v + randn() * self.v_noise_std,
                self.a)

    def step_and_sense(self):
        self.a      += randn() * self.a_std;
        self.v      += self.a*self.dt
        self.x      += self.v*self.dt

        return (self.x + randn() * self.x_noise_std,
                self.v + randn() * self.v_noise_std,
                self.x, self.v, self.a)

    def step_and_track(self, n):
        x,xDot,t_x,t_v,a = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'xDot' : xDot, 'x.true' : t_x, 'xDot.true' : t_v, 'a' : a } )
# ---------------------------------------------------------------------------------
class PseudoConst2DAccelerationSensor(object):
    ''' Simulate an object moving at constant acceleration'''
    def __init__(self, x=(0,0), v=(1,0), a=(1,0), dt=0.1, x_noise_std=1., v_noise_std=0., a_std=0. ):
        self.x_sensor = PseudoConst1DAccelerationSensor( x=x[0], v=v[0], a=a[0], dt=dt, x_noise_std=x_noise_std, v_noise_std=v_noise_std, a_std=a_std )
        self.y_sensor = PseudoConst1DAccelerationSensor( x=x[1], v=v[1], a=a[1], dt=dt, x_noise_std=x_noise_std, v_noise_std=v_noise_std, a_std=a_std )

    def step(self):
        x,xv,xa = self.x_sensor.step()
        y,yv,ya = self.y_sensor.step()
        return (x,y, xv,yv, xa,ya)

    def sense(self):
        x,xv,xa = self.x_sensor.sense()
        y,yv,ya = self.y_sensor.sense()
        return (x,y, xv,yv, xa,ya)

    def step_and_sense(self):
        x,xv,t_x,t_xv,xa = self.x_sensor.step_and_sense()
        y,yv,t_y,t_yv,ya = self.y_sensor.step_and_sense()
        return (x,y, xv,yv, t_x,t_y, t_xv, t_yv, xa,ya)

    def step_and_track(self, n):
        x,y, xv,yv, t_x,t_y, t_xv, t_yv, xa,ya =  zip( *[self.step_and_sense() for _ in range(n)] )
        return DataFrame( { 'x'         :    x, 'y'         :    y,
                            'xDot'      :   xv, 'yDot'      :   yv,
                            'x.true'    :  t_x, 'y.true'    :  t_y,
                            'xDot.true' : t_xv, 'yDot.true' : t_yv,
                            'ax'        :   xa, 'ay'        :   ya
                           } )
# =================================================================================
class Sinusoidal1DSensor(object):
    ''' Simulate an object moving at constant angular velocity'''
    def __init__(self, phase=0, x0=0, amplitude=1., dt=0.1, freq=1., noise_std=1.):
        self.amplitude = amplitude
        self.phase     = float(phase)
        self.noise_std = float(noise_std)
        self.dt        = float(dt)
        self.dtheta    = 2. * pi * freq * self.dt
        self.x0        = x0;
        self.x         = x0 + amplitude*sin(phase)

    def step(self):
        self.phase += self.dtheta
        self.x      = self.x0+self.amplitude*sin( self.phase )
        return self.x

    def sense(self):
        return self.x + randn() * self.noise_std

    def step_and_sense(self):
        self.phase += self.dtheta
        self.x      = self.x0 + self.amplitude*sin( self.phase )
        return ( self.x + randn() * self.noise_std, self.x )

    def step_and_track(self, n):
        x,t_x = zip( *[self.step_and_sense() for _ in range(n)])
        return DataFrame( { 'x' : x, 'x.true' : t_x } )

