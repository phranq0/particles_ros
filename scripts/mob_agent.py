# Random number generator for simulating noisy measurements
import numpy as np

# Class for defining a simple simulated mobile robot (holonomic planar case)

class MobAgent:
    # Robot state
    x = 0
    y = 0
    # Simulation step
    dt = 0   

    # Constructor
    def __init__(self,x_init,y_init,step):
        self.x = x_init
        self.y = y_init 
        self.dt = step

    # Advance state
    def step(self,vx,vy):
        self.x += vx*self.dt
        self.y += vy*self.dt

    # Simulate direct state noisy measurements 
    def state_meas(self):
        mu = 0
        sigma = 0.05
        x_m = self.x + np.random.normal(mu,sigma)
        y_m = self.y + np.random.normal(mu,sigma)
        #print("Measurements: %f and %f" % (x_m,y_m))
        return [x_m,y_m]