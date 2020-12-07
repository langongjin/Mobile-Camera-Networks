import numpy as np


class Config:
    def __init__(self):
        self.PROJECT_NAME = "Mobile Camera Networks"

        self.sample_time = 0.2  # time interval to update human states
        self.time_count = 1   # represent current time when save figures

        # camera parameters
        self.radius = 4.0
        self.fov = 60.0
        self.cam_number = 4
        self.cam_id = [0, 1, 2, 3]
        self.cam_location = np.array([[4.0, 0.0], [0.0, 4.0], [4.0, 8.0], [8.0, 4.]])
        self.cam_focal = np.array([90.0, 0.0, -90.0, 180.0])
        self.cam_colors = np.linspace(0, 1, 4)

        self.x_ticks = np.arange(0, 8, 1)
        self.y_ticks = np.arange(0, 8, 1)

        self.KC = np.array([[150.0, 120.0, 90.0, 60.0, 30.0],
                            [60.0, 30.0, 0.0, -30.0, -60.0],
                            [-150.0, -120.0, -90.0, -60.0, -30.0],
                            [120.0, 150.0, 180.0, -150.0, -120.0]])

        self.move = np.array([-0.15, 0, 0.15])

        self.delta_k = 0.02

        self.colors = ['red', 'blue', 'black', 'green', 'cyan', 'magenta', 'yellow']
        self.markers = ['o', '*', 'x', 'v', '^', 's', '+', '>', '<', 's', 'p']






