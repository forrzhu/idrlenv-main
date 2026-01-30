__credits__ = ["Andrea PIERRÉ, weipeng0098@126.com"]

import math
from typing import Optional

import numpy as np
import pygame
from pygame import gfxdraw
import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)
import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from itertools import chain





FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape = polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    # Create a polygon shape with dimensions specified as half-width and half-height.
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if contact.fixtureA.body in [robot.hull for robot in self.env.robots] or \
            contact.fixtureB.body in [robot.hull for robot in self.env.robots]:
            self.env.game_over = True
        for leg in sum([[robot.legs[1], robot.legs[3]] for robot in self.env.robots],[]):
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in sum([[robot.legs[1], robot.legs[3]] for robot in self.env.robots],[]):
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

class SingleWalker:
    def __init__(self):
        self.hull: Optional[Box2D.b2Body] = None
        self.legs: Optional[Box2D.b2Body] = None

class BipedalWalkerAgg(gym.Env, EzPickle):
    """
    ## Description
    This is a aggregated 4-joint walker robot environment.
    Multiple simple 4-joint walker robot are connected by revolt joints.
    There are two versions:
    - Normal, with slightly uneven terrain.
    - Hardcore, with ladders, stumps, pitfalls.

    To solve the normal version, you need to get 300 points in 1600 time steps.
    To solve the hardcore version, you need 300 points in 2000 time steps.

    A heuristic is provided for testing. It's also useful to get demonstrations
    to learn from. To run the heuristic:
    ```
    python gymnasium/envs/box2d/bipedal_walker.py
    ```

    ## Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4*n_contents joints at both hips and knees.

    ## Observation Space
    State consists of hull angle speed, angular velocity, horizontal speed,
    vertical speed, position of joints and joints angular speed, legs contact
    with ground, and 10 lidar rangefinder measurements. There are no coordinates
    in the state vector.

    ## Rewards
    Reward is given for moving forward, totaling 300+ points up to the far end.
    If the robot falls, it gets -100. Applying motor torque costs a small
    amount of points. A more optimal agent will get a better score.

    ## Starting State
    The walker starts standing at the left end of the terrain with the hull
    horizontal, and both legs in the same position with a slight knee angle.

    ## Episode Termination
    The episode will terminate if the hull gets in contact with the ground or
    if the walker exceeds the right end of the terrain length.

    ## Arguments
    To use to the _hardcore_ environment, you need to specify the
    `hardcore=True` argument like below:
    ```python
    import gymnasium as gym
    env = gym.make("BipedalWalker-v3", hardcore=True)
    ```

    ## Version History
    - v3: Returns the closest lidar trace instead of furthest;
        faster video recording
    - v2: Count energy spent
    - v1: Legs now report contact with ground; motors have higher torque and
        speed; ground has higher friction; lidar rendered less nervously.
    - v0: Initial version


    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, n_contents, render_mode: Optional[str] = None, hardcore: bool = False):
        EzPickle.__init__(self, n_contents)
        self.screen = None
        self.clock=None
        self.isopen = True
        
        self.n_contents = n_contents
        self.world = Box2D.b2World()
        self.terrain = None
        
        self.robots= []
        self.connections = []

        self.prev_shaping = None

        self.hardcore = hardcore

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
        low = np.array(
            [
                -math.pi,
                -5.0,
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -math.pi,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
            ]*self.n_contents
            + [-1.0] * 10
        ).astype(np.float32)
        high = np.array(
            [
                math.pi,
                5.0,
                5.0,
                5.0,
                math.pi,
                5.0,
                math.pi,
                5.0,
                5.0,
                math.pi,
                5.0,
                math.pi,
                5.0,
                5.0,
            ]*self.n_contents
            + [1.0] * 10
        ).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]*self.n_contents).astype(np.float32),
            np.array([1, 1, 1, 1]*self.n_contents).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        for robot in self.robots:
            self.world.DestroyBody(robot.hull)
            for leg in robot.legs:
                self.world.DestroyBody(leg)
        self.robots = []
        self.connections = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        stair_steps, stair_width, stair_height = 0, 0, 0
        original_y = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.random() > 0.5 else -1
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (
                    x
                    + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                    y
                    + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                )
                for a in range(5)
            ]
            x1 = min(p[0] for p in poly)
            x2 = max(p[0] for p in poly)
            self.cloud_poly.append((poly, x1, x2))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0
        self.step_counter=0

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        self.robots = [self.createRobot(i) for i in range(self.n_contents)]
        self.createConnection()
        self.drawlist = self.terrain \
                            + sum([robot.legs for robot in self.robots], []) \
                            + [robot.hull for robot in self.robots]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0]*self.n_contents))[0], {}

    def stepOneRobot(self, robot, action, control_speed):
        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
          # Should be easier as well
        if control_speed:
            robot.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            robot.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            robot.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            robot.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            robot.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            robot.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            robot.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            robot.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            robot.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            robot.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            robot.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            robot.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )

    def composeState(self):
        states = []
        for robot in self.robots:
            vel = robot.hull.linearVelocity
            state = [
                robot.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
                2.0 * robot.hull.angularVelocity / FPS,
                0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
                0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
                robot.joints[0].angle,
                # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
                robot.joints[0].speed / SPEED_HIP,
                robot.joints[1].angle + 1.0,
                robot.joints[1].speed / SPEED_KNEE,
                1.0 if robot.legs[1].ground_contact else 0.0,
                robot.joints[2].angle,
                robot.joints[2].speed / SPEED_HIP,
                robot.joints[3].angle + 1.0,
                robot.joints[3].speed / SPEED_KNEE,
                1.0 if robot.legs[3].ground_contact else 0.0,
            ]
            states.append(state)
        return sum(states, [])
        
    def step(self, action: np.ndarray):
        assert len(self.robots)>0

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        action_sep = action.reshape(self.n_contents, -1)
        for robot, act in zip(self.robots, action_sep):
            self.stepOneRobot(robot, act, control_speed)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.robots[-1].hull.position
        

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
        state = self.composeState()
        
        state += [l.fraction for l in self.lidar]
        assert len(state) == 14*self.n_contents+10

        self.scroll = self.robots[0].hull.position.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        # shaping -= 5.0 * abs(
        #     state[0]
        # )  
        # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        terminated = False
        if self.game_over or pos[0] < 0:
            # reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True
        self.step_counter+=1
        if self.render_mode == "human":
            self.render()
        if self.step_counter==1000:
            terminated=True
        return np.array(state, dtype=np.float32), reward, terminated, False, {}
        # return np.array(state, dtype=np.float32), reward, done, {}


    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, single_lidar.p1[1] * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, single_lidar.p2[1] * SCALE),
                    width=1,
                )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        flagy1 = TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        x = TERRAIN_STEP * 3 * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def createRobot(self, i):
        robot = SingleWalker()
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2 +2*LEG_H*i
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        robot.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        robot.hull.color1 = (127, 51, 229)
        robot.hull.color2 = (76, 76, 127)
        robot.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        robot.legs: List[Box2D.b2Body] = []
        robot.joints: List[Box2D.b2RevoluteJoint] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=robot.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            robot.legs.append(leg)
            robot.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            robot.legs.append(lower)
            robot.joints.append(self.world.CreateJoint(rjd))
        return robot

    def createConnection(self):
        for i, robot in enumerate(self.robots[:-1]):
            rjd = revoluteJointDef(
                bodyA=self.robots[i].hull,
                bodyB=self.robots[i+1].hull,
                localAnchorA=((+34/SCALE, 0/SCALE)),
                localAnchorB=(-30/SCALE, 0/SCALE),
                enableMotor=False,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.,
                upperAngle=1.,
            )
            self.world.CreateJoint(rjd)
            self.connections.append(rjd)

class BipedalWalkerHardcore:
    def __init__(self):
        raise error.Error(
            "Error initializing BipedalWalkerHardcore Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the hardcore keyword in gym.make, i.e.\n"
            'gym.make("BipedalWalker-v3", hardcore=True)'
        )


def generate_action_for_one(single_state):
    contact0 = single_state[8]
    contact1 = single_state[13]
    
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    
    moving_s_base = 4 + 5 * moving_leg
    supporting_s_base = 4 + 5 * supporting_leg

    hip_targ = [None, None]  # -0.8 .. +1.1
    knee_targ = [None, None]  # -0.6 .. +0.9
    hip_todo = [0.0, 0.0]
    knee_todo = [0.0, 0.0]

    if state == STAY_ON_ONE_LEG:
        hip_targ[moving_leg] = 1.1
        knee_targ[moving_leg] = -0.6
        supporting_knee_angle += 0.03
        if single_state[2] > SPEED:
            supporting_knee_angle += 0.03
        supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
        knee_targ[supporting_leg] = supporting_knee_angle
        if single_state[supporting_s_base + 0] < 0.10:  # supporting leg is behind
            state = PUT_OTHER_DOWN
    if state == PUT_OTHER_DOWN:
        hip_targ[moving_leg] = +0.1
        knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
        knee_targ[supporting_leg] = supporting_knee_angle
        if single_state[moving_s_base + 4]:
            state = PUSH_OFF
            supporting_knee_angle = min(single_state[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
    if state == PUSH_OFF:
        knee_targ[moving_leg] = supporting_knee_angle
        knee_targ[supporting_leg] = +1.0
        if single_state[supporting_s_base + 2] > 0.88 or single_state[2] > 1.2 * SPEED:
            state = STAY_ON_ONE_LEG
            moving_leg = 1 - moving_leg
            supporting_leg = 1 - moving_leg

    if hip_targ[0]:
        hip_todo[0] = 0.9 * (hip_targ[0] - single_state[4]) - 0.25 * single_state[5]
    if hip_targ[1]:
        hip_todo[1] = 0.9 * (hip_targ[1] - single_state[9]) - 0.25 * single_state[10]
    if knee_targ[0]:
        knee_todo[0] = 4.0 * (knee_targ[0] - single_state[6]) - 0.25 * single_state[7]
    if knee_targ[1]:
        knee_todo[1] = 4.0 * (knee_targ[1] - single_state[11]) - 0.25 * single_state[12]

    hip_todo[0] -= 0.9 * (0 - single_state[0]) - 1.5 * single_state[1]  # PID to keep head strait
    hip_todo[1] -= 0.9 * (0 - single_state[0]) - 1.5 * single_state[1]
    knee_todo[0] -= 15.0 * single_state[3]  # vertical speed, to damp oscillations
    knee_todo[1] -= 15.0 * single_state[3]
    a = [hip_todo[0], knee_todo[0], hip_todo[1], knee_todo[1]]
    return a

if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    n_content=4
    env = BipedalWalkerAgg(render_mode='human', n_contents=n_content)
    env.reset()
    steps = 0
    total_reward = 0
    a = np.zeros(4*n_content)
    while True:
        s, r, terminated, truncated, info = env.step(a)

        total_reward += r
        if steps % 20 == 0 or terminated or truncated:
            print("\naction " + str([f"{x:+0.2f}" for x in a]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
            print("hull " + str([f"{x:+0.2f}" for x in s[0:4]]))
            print("leg0 " + str([f"{x:+0.2f}" for x in s[4:9]]))
            print("leg1 " + str([f"{x:+0.2f}" for x in s[9:14]]))
        steps += 1
        
        actions = [generate_action_for_one(s[i:i + 14]) for i in range(0, 14 * n_content, 14)]
        a = np.concatenate(actions)
        a = np.clip(0.5 * a, -1.0, 1.0)

        if terminated or truncated:
            break
