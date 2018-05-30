"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pygame 

logger = logging.getLogger(__name__)

class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.righ_key_pressed = False
        self.left_key_pressed = False
        self.key = -1 
        self.video_size = 200, 200 
        self.screen = pygame.display.set_mode(self.video_size) 
        self.clock = pygame.time.Clock() 
        self.fps = 50 
        self.min_action = -1.0 
        self.max_action = 1.0 
        self.h = 0  # Human correction (reward)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 40 * 2 * math.pi / 360  
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,)) 
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * action # Continuous force
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state), np.array([self.h, reward]), done, {}  # Now return human feedback and reward value

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    # Capture key from keyboard for COACH correction
    def capture_key(self):
        # process pygame events
        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                if(event.key == 275):
                    pygame.display.flip()
                    self.clock.tick(self.fps)
                    #print('right key pressed', event.key)
                    return 'rightkeydown'
                elif(event.key == 276):
                    pygame.display.flip()
                    self.clock.tick(self.fps)
                    #print('left key pressed', event.key)
                    return 'leftkeydown'
            if event.type == pygame.KEYUP:
                if(event.key == 275):
                    pygame.display.flip()
                    self.clock.tick(self.fps)
                    return 'rightkeyup'
                elif(event.key == 276):
                    pygame.display.flip()
                    self.clock.tick(self.fps)
                    return 'leftkeyup'

        pygame.display.flip()
        self.clock.tick(self.fps)
        return -1

    def get_feedback(self):
        return self.h

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])

            ### Generate right and left arrows when COACH correction received
            rightarrow = rendering.FilledPolygon([(l/2,b/1.5), (l/2,t/1.5), (r,(t/1.5+b/1.5)/2)])
            self.rightarrowtrans = rendering.Transform()
            rightarrow.add_attr(self.rightarrowtrans)
            rightarrow.set_color(0.2,0.4,0.8)
            self.viewer.add_geom(rightarrow)
            self.rightarrowtrans.set_translation(1000, 1000)

            leftarrow = rendering.FilledPolygon([(r+l/2,b/1.5), (r+l/2,t/1.5), (-r,(t/1.5+b/1.5)/2)])
            self.leftarrowtrans = rendering.Transform()
            leftarrow.add_attr(self.leftarrowtrans)
            leftarrow.set_color(0.2,0.4,0.8)
            self.viewer.add_geom(leftarrow)
            self.leftarrowtrans.set_translation(1000, 1000)
            ###

            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART

        ##: capture key and show arrow in canvas       
        self.key = self.capture_key()

        if self.key == 'rightkeyup':
            self.righ_key_pressed = False
            self.rightarrowtrans.set_translation(1000, 1000)
            self.h = 0
        if self.key == 'leftkeyup':
            self.left_key_pressed = False
            self.leftarrowtrans.set_translation(1000, 1000)
            self.h = 0
        if self.righ_key_pressed:
            self.rightarrowtrans.set_translation(cartx+40, carty+30)
            self.h = 1
        if self.left_key_pressed:
            self.leftarrowtrans.set_translation(cartx-40, carty+30)
            self.h = -1
        if self.key == 'rightkeydown':
            self.righ_key_pressed = True
            self.h = 1 # go right
            self.rightarrowtrans.set_translation(cartx+40, carty+30)
            self.leftarrowtrans.set_translation(1000, 1000)
        if self.key == 'leftkeydown':
            self.left_key_pressed = True
            self.h = -1 # go left
            self.leftarrowtrans.set_translation(cartx-40, carty+30)
            self.rightarrowtrans.set_translation(1000, 1000)             
        ###

        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
