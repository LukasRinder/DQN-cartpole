import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CartPoleModelParameters():
    def __init__(self):
        self.g = 9.82
        self.l = 0.6
        self.m_c = 0.5
        self.m_p = 0.5
        self.b = 0.1

        self.force_limit = 10.0

        self.dtheta_limits = np.array([-10.0, 10.0])
        self.x_limits = np.array([-6.0, 6.0])
        self.dx_limits = np.array([-10, 10])
        self.th_limits = np.array([-np.pi, np.pi])


class CartPoleModel():
    def __init__(self, dt_sim, dt_action, t_abort, model_params=None):
        self.p = model_params or CartPoleModelParameters()

        self.sim_steps_per_action = int(dt_action / dt_sim)
        self.dt_action = dt_action

        self.t_abort = t_abort

        self.dt_sim = dt_sim
        self.dt_sim_2 = dt_sim**2

        self.x_target = self.p.x_limits * 0.05  # goal state: x = 0
        self.th_target = self.p.th_limits * (1-0.05)  # goal state: th = +/- pi

        self.target_range = False
        self.steady_state_time = 0
        self.first_target_range = False

        self.x_overshoot = 0
        self.th_overshoot = 0

        self.task_accomplished = False  # task accomplished if pendulum is stable in target range for 1 s

    def reset_env(self, std=0.3):
        pos_down = np.array([0.0, 0.0, 0.0, 0.0])
        pos_down_std = np.array([std, std, std, std])

        self.state = np.random.normal(pos_down, pos_down_std)
        self.t = 0

        self.target_range = False
        self.steady_state_time = 0
        self.first_target_range = False

        self.x_overshoot = 0
        self.th_overshoot = 0

        self.task_accomplished = False

        return self.state

    def get_state(self):
        return self.state

    def step(self, force, f_disturbance=0):
        force = min(max(force, -self.p.force_limit), self.p.force_limit)

        force = force + f_disturbance

        x, dx, theta, dtheta = self.state

        # Simulate CartPole
        for i in range(self.sim_steps_per_action):
            s = np.sin(theta)
            c = np.cos(theta)

            ddx = (2*self.p.m_p*self.p.l*dtheta**2*s + 3*self.p.m_p*self.p.g*s*c + 4*force - 4*self.p.b*dx)/(4*(self.p.m_p + self.p.m_c) - 3*self.p.m_p*c**2)

            ddtheta = (-3*self.p.m_p*self.p.l*dtheta**2*s*c - 6*(self.p.m_p + self.p.m_c)*self.p.g*s - 6*(force - self.p.b*dx)*c)/(4*self.p.l*(self.p.m_p + self.p.m_c) - 3*self.p.m_p*self.p.l*c**2)

            dx = dx + ddx*self.dt_sim
            dtheta = dtheta + ddtheta*self.dt_sim

            # Limit dtheta und dx
            dtheta = min(max(dtheta, self.p.dtheta_limits[0]), self.p.dtheta_limits[1])
            dx = min(max(dx, self.p.dx_limits[0]), self.p.dx_limits[1])

            x = x + dx*self.dt_sim + 0.5*self.dt_sim_2*ddx
            theta = theta + dtheta*self.dt_sim + 0.5*self.dt_sim_2*ddtheta

            # Avoid theta overflow
            if theta > np.pi:
                theta = theta - 2 * np.pi
            elif theta < -np.pi:
                theta = theta + 2 * np.pi

            # enforce limits of x
            x = min(max(x, self.p.x_limits[0]), self.p.x_limits[1])

        t = self.t
        self.t += self.dt_action

        # Abort if time exceeded or cart out of limits
        abort = x <= self.p.x_limits[0] or x >= self.p.x_limits[1] or self.t >= self.t_abort

        self.state = (x, dx, theta, dtheta)

        self.check_target_range()

        # Calculate Reward
        e = -0.5*(x**2 + 2*self.p.l**2*(1 + np.cos(theta)) + 2*x*self.p.l*np.sin(theta))
        reward = -(1-np.exp(e))

        return t, self.state, reward, abort

    def check_target_range(self):
        # check if state inside target range, and set "target_range" accordingly
        if (self.x_target[0] <= self.state[0] <= self.x_target[1]) and (abs(self.state[2]) >= self.th_target[1]):
            if not self.target_range:
                self.target_range = True
                self.first_target_range = True
                if not self.task_accomplished:
                    self.steady_state_time = self.t
                # print(f"Target range met with steady state time: {self.steady_state_time}")
            else:
                if self.t - self.steady_state_time >= 1 and not self.task_accomplished:
                    self.task_accomplished = True

        else:
            self.target_range = False
            if not self.task_accomplished:
                self.steady_state_time = 0
                # print(f"Out of target range again. Resetting steady state time.")

        # save the maximal overshoot after the target range was met for the first time
        if self.first_target_range:
            # check overshoot in x
            if self.x_overshoot < abs(self.state[0]):
                self.x_overshoot = self.state[0]

            # check overshoot in theta
            if self.th_overshoot < (np.pi - abs(self.state[2])):
                self.th_overshoot = np.pi - abs(self.state[2])

    def visualize(self, s_traj, name="im"):
        """
        Visualize the cart pole and save a video.
        :param s_traj: Trajectory of states [dth, th, dx, x] for several time steps, e.g. shape (N, 4).
        """

        # create writer for video
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=-1)

        # create figure
        fig = plt.figure(figsize=(7, 1.4))

        ax = fig.add_subplot(111, aspect='equal', xlim=(-3.5, 3.5), ylim=(-0.7, 0.7))
        ax.grid()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_yaxis().set_ticks([0])
        ax.get_xaxis().set_ticks([-1, 0, 1])

        # objects for cart, pole, polehead, and time
        pole, = ax.plot([], [], '-', lw=2, c='b')
        cart, = ax.plot([], [], 's', ms=7, c='b')
        time_template = '%.1f s'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            """initialize animation"""
            pole.set_data([], [])
            cart.set_data([], [])
            time_text.set_text('')

            return pole, cart, time_text

        def animate(i):
            """perform animation step"""
            x_cart = s_traj[i, 0]
            y_cart = 0

            x_pole = np.sin(s_traj[i, 2]) * self.p.l + x_cart
            y_pole = - np.cos(s_traj[i, 2]) * self.p.l

            cart.set_data([x_cart], [y_cart])
            pole.set_data([x_cart, x_pole], [y_cart, y_pole])  # line with start and end coordinates
            time_text.set_text(time_template % (i * self.dt_action))

            return pole, cart, time_text

        ani = animation.FuncAnimation(fig, animate, frames=len(s_traj), interval=self.dt_action, blit=True, init_func=init)
        ani.save(name + ".mp4", writer=writer, dpi=300)
