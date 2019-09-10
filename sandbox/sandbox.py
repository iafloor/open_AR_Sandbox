import os
from warnings import warn

# class infrastructre
from abc import ABCMeta, abstractmethod

try:
    import freenect

    warn('Two kernels cannot access the kinect at the same time. This will lead to a sudden death of the kernel. ' \
         'Be sure no other kernel is running before initialize a kinect object.', RuntimeWarning)
except ImportError:
    warn(
        'Freenect is not installed. if you are using the Kinect Version 2 on a windows machine, use the KinectV2 class!')

try:
    from pykinect2 import PyKinectV2  # try to import Wrapper for KinectV2 Windows SDK
    from pykinect2 import PyKinectRuntime

except ImportError:
    pass

try:
    import cv2
    from cv2 import aruco

except ImportError:
    # warn('opencv is not installed. Object detection will not work')
    pass

import webbrowser
import pickle
import numpy
import scipy
import scipy.ndimage

# for new projector
import panel as pn

# for DummySensor
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata

from itertools import count
from PIL import Image, ImageDraw
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
# import gempy.hackathon as hackathon
import IPython
import threading

import json
import pandas as pd

# TODO: When we move GeoMapModule import gempy just there
import gempy as gp


class Kinect:
    """
    Masterclass for initializing the kinect.
    Init the kinect and provides a method that returns the scanned depth image as numpy array. Also we do the gaussian
    blurring to get smoother lines.
    """

    def __init__(self, calibrationdata, filter='gaussian', n_frames=3, sigma_gauss=3):

        self.calib = calibrationdata
        self.calib.s_name = self.name
        self.calib.s_width = self.depth_width
        self.calib.s_height = self.depth_height

        self.id = None
        self.device = None
        self.angle = None

        self.depth = None
        self.color = None
        self.ir_frame_raw = None
        self.ir_frame = None


        # TODO: include filter self.-filter parameters as function defaults
        self.filter = filter # TODO: deprecate get_filtered_frame, make it switchable in runtime
        self.n_frames = n_frames  # filter parameters
        self.sigma_gauss = sigma_gauss

        self.setup()

    def get_filtered_frame(self):

        # collect last n frames in a stack
        depth_array = self.get_frame()
        for i in range(self.n_frames - 1):
            depth_array = numpy.dstack([depth_array, self.get_frame()])
        # calculate mean values ignoring zeros by masking them
        depth_array_masked = numpy.ma.masked_where(depth_array == 0, depth_array) # needed for V2?
        self.depth = numpy.ma.mean(depth_array_masked, axis=2)
        # apply gaussian filter
        self.depth = scipy.ndimage.filters.gaussian_filter(self.depth, self.sigma_gauss)

        return self.depth


class KinectV1(Kinect):

    # hard coded class attributes for KinectV1's native resolution
    name = 'kinect_v1'
    depth_width = 320
    depth_height = 240
    color_width = 640
    color_height = 480
    # TODO: Check!

    def setup(self):
        print("looking for kinect...")
        ctx = freenect.init()
        self.device = freenect.open_device(ctx, self.id)
        print(self.id)
        freenect.close_device(self.device)  # TODO Test if this has to be done!
        # get the first Depth frame already (the first one takes much longer than the following)
        self.depth = self.get_frame()
        print("kinect initialized")

    def set_angle(self, angle):  # TODO: throw out
        """
        Args:
            angle:

        Returns:
            None
        """
        self.angle = angle
        freenect.set_tilt_degs(self.device, self.angle)

    def get_frame(self):
            self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
            self.depth = numpy.fliplr(self.depth)
            return self.depth

    def get_rgb_frame(self):  # TODO: check if this can be thrown out
        """

        Returns:

        """
        self.color = freenect.sync_get_video(index=self.id)[0]
        self.color = numpy.fliplr(self.color)
        return self.color

    def calibrate_frame(self, frame, calibration=None):  # TODO: check if this can be thrown out
        """

        Args:
            frame:
            calibration:

        Returns:

        """
        if calibration is None:
            print("no calibration provided!")
        rotated = scipy.ndimage.rotate(frame, calibration.calibration_data.rot_angle, reshape=False)
        cropped = rotated[calibration.calibration_data.y_lim[0]: calibration.calibration_data.y_lim[1],
                  calibration.calibration_data.x_lim[0]: calibration.calibration_data.x_lim[1]]
        cropped = numpy.flipud(cropped)
        return cropped


class KinectV2(Kinect):
    """
    control class for the KinectV2 based on the Python wrappers of the official Microsoft SDK
    Init the kinect and provides a method that returns the scanned depth image as numpy array.
    Also we do gaussian blurring to get smoother surfaces.

    """

    # hard coded class attributes for KinectV2's native resolution
    name = 'kinect_v2'
    depth_width = 512
    depth_height = 424
    color_width = 1920
    color_height = 1080

    def setup(self):
        self.device = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.depth = self.get_frame()
        self.color = self.get_color()
        #self.ir_frame_raw = self.get_ir_frame_raw()
        #self.ir_frame = self.get_ir_frame()

    def get_frame(self):
        """

        Args:

        Returns:
               2D Array of the shape(424, 512) containing the depth information of the latest frame in mm

        """
        depth_flattened = self.device.get_last_depth_frame()
        self.depth = depth_flattened.reshape(
            (self.depth_height, self.depth_width))  # reshape the array to 2D with native resolution of the kinectV2
        return self.depth

    def get_ir_frame_raw(self):
        """

        Args:

        Returns:
               2D Array of the shape(424, 512) containing the raw infrared intensity in (uint16) of the last frame

        """
        ir_flattened = self.device.get_last_infrared_frame()
        self.ir_frame_raw = numpy.flipud(
            ir_flattened.reshape((self.depth_height, self.depth_width)))  # reshape the array to 2D with native resolution of the kinectV2
        return self.ir_frame_raw

    def get_ir_frame(self, min=0, max=6000):
        """

        Args:
            min: minimum intensity value mapped to uint8 (will become 0) default: 0
            max: maximum intensity value mapped to uint8 (will become 255) default: 6000
        Returns:
               2D Array of the shape(424, 512) containing the infrared intensity between min and max mapped to uint8 of the last frame

        """
        ir_frame_raw = self.get_ir_frame_raw()
        self.ir_frame = numpy.interp(ir_frame_raw, (min, max), (0, 255)).astype('uint8')
        return self.ir_frame

    def get_color(self):
        color_flattened = self.device.get_last_color_frame()
        resolution_camera = self.color_height * self.color_width  # resolution camera Kinect V2
        # Palette of colors in RGB / Cut of 4th column marked as intensity
        palette = numpy.reshape(numpy.array([color_flattened]), (resolution_camera, 4))[:, [2, 1, 0]]
        position_palette = numpy.reshape(numpy.arange(0, len(palette), 1), (self.color_height, self.color_width))
        self.color = numpy.flipud(palette[position_palette])
        return self.color

class DummySensor:

    def __init__(self, calibrationdata, width=512, height=424, depth_limits=(80, 100), points_n=5, points_distance=20,
                 alteration_strength=0.1, random_seed=None):

        # alteration_strength: 0 to 1 (maximum 1 equals numpy.pi/2 on depth range)

        self.calib = calibrationdata

        self.depth_width = width
        self.depth_height = height
        # update calibration according to device
        self.calib.s_name = 'dummy'
        self.calib.s_width = self.depth_width
        self.calib.s_height = self.depth_height

        self.depth_lim = depth_limits
        self.n = points_n
        self.distance = points_distance
        self.strength = alteration_strength
        self.seed = random_seed

        # create grid, init values, and init interpolation
        self.grid = self.create_grid()
        self.positions = self.pick_positions()

        self.os_values = None
        self.values = None
        self.pick_values()

        self.interpolation = None
        self.interpolate()

    ## Methods

    def get_frame(self):
        # TODO: Add time check for 1/30sec
        self.alter_values()
        self.interpolate()
        return self.interpolation

    def get_filtered_frame(self):
        return self.get_frame()

    ## Private functions
    # TODO: Make private

    def oscillating_depth(self, random):
        r = (self.depth_lim[1] - self.depth_lim[0]) / 2
        return numpy.sin(random) * r + r + self.depth_lim[0]

    def create_grid(self):
        # creates 2D grid for given resolution
        x, y = numpy.meshgrid(numpy.arange(0, self.depth_width, 1), numpy.arange(0, self.depth_height, 1))
        return numpy.stack((x.ravel(), y.ravel())).T

    def pick_positions(self, corners=True, seed=None):
        '''
        grid: Set of possible points to pick from
        n: desired number of points, not guaranteed to be reached
        distance: distance or range, pilot points should be away from dat points
        '''

        numpy.random.seed(seed=seed)

        gl = self.grid.shape[0]
        gw = self.grid.shape[1]
        points = numpy.zeros((self.n, gw))

        # randomly pick initial point
        ipos = numpy.random.randint(0, gl)
        points[0, :2] = self.grid[ipos, :2]

        i = 1  # counter
        while i < self.n:

            # calculate all distances between remaining candidates and sim points
            dist = cdist(points[:i, :2], self.grid[:, :2])
            # choose candidates which are out of range
            mm = numpy.min(dist, axis=0)
            candidates = self.grid[mm > self.distance]
            # count candidates
            cl = candidates.shape[0]
            if cl < 1: break
            # randomly pick candidate and set next pilot point
            pos = numpy.random.randint(0, cl)
            points[i, :2] = candidates[pos, :2]

            i += 1

        # just return valid points if early break occured
        points = points[:i]

        if corners:
            c = numpy.zeros((4, gw))
            c[1, 0] = self.grid[:, 0].max()
            c[2, 1] = self.grid[:, 1].max()
            c[3, 0] = self.grid[:, 0].max()
            c[3, 1] = self.grid[:, 1].max()
            points = numpy.vstack((c, points))

        return points

    def pick_values(self):
        numpy.random.seed(seed=self.seed)
        n = self.positions.shape[0]
        self.os_values = numpy.random.uniform(-numpy.pi, numpy.pi, n)
        self.values = self.oscillating_depth(self.os_values)

    def alter_values(self):
        # maximum range in both directions the values should be altered
        numpy.random.seed(seed=self.seed)
        os_range = self.strength * (numpy.pi / 2)
        for i, value in enumerate(self.os_values):
            self.os_values[i] = value + numpy.random.uniform(-os_range, os_range)
        self.values = self.oscillating_depth(self.os_values)

    def interpolate(self):
        inter = griddata(self.positions[:, :2], self.values, self.grid[:, :2], method='cubic', fill_value=0)
        self.interpolation = inter.reshape(self.depth_height, self.depth_width)


class Calibration:
    """
    TODO:refactor completely! Make clear distinction between the calibration methods and calibration Data!
    Tune calibration parameters. Save calibration file. Have methods to project so we can see what we are calibrating
    """

    def __init__(self, calibrationdata, sensor, projector):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector

        self.frame = self.sensor.get_filtered_frame()

        self.fig = plt.figure()
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.fig.add_axes(self.ax)
        self.plot_sensor()

        pn.extension()
        self.pn_fig = pn.pane.Matplotlib(self.fig, tight=False)


    def calibrate_projector(self):

        margin_top = pn.widgets.IntSlider(name='Top margin', value=self.calib.p_frame_top, start=0, end=200)
        def callback_mt(target, event):
            m = target.margin
            n = event.new
            # just changing single indices does not trigger updating of pane
            target.margin = [n, m[1], m[2], m[3]]
            # also update calibration object
            self.calib.p_frame_top = event.new

        margin_top.link(self.projector.frame, callbacks={'value': callback_mt})

        margin_left = pn.widgets.IntSlider(name='Left margin', value=self.calib.p_frame_left, start=0, end=200)
        def callback_ml(target, event):
            m = target.margin
            n = event.new
            # just changing single indices does not trigger updating of pane
            target.margin = [m[0], m[1], m[2], n]
            # also update calibration object
            self.calib.p_frame_left = event.new
        margin_left.link(self.projector.frame, callbacks={'value': callback_ml})

        width = pn.widgets.IntSlider(name='Map width', value=self.calib.p_frame_width, start=self.calib.p_frame_width - 400, end=self.calib.p_frame_width + 800)
        def callback_width(target, event):
            target.width = event.new
            target.param.trigger('object')
            # also update calibration object
            self.calib.p_frame_width = event.new
        width.link(self.projector.frame, callbacks={'value': callback_width})

        height = pn.widgets.IntSlider(name='Map height', value=self.calib.p_frame_height, start=self.calib.p_frame_height - 400, end=self.calib.p_frame_height + 800)
        def callback_height(target, event):
            target.height = event.new
            target.param.trigger('object')
            # also update calibration object
            self.calib.p_frame_height = event.new
            #self.plot.redraw_plot()
        height.link(self.projector.frame, callbacks={'value': callback_height})

        widgets = pn.Column("### Map positioning", margin_top, margin_left, width, height)
        return widgets


    def calibrate_sensor(self):
        def callback_mst(target, event):
            # set value in calib
            self.calib.s_top = event.new
            # change plot and trigger panel update
            self.plot_sensor()
            self.pn_fig.param.trigger('object')

        def callback_msr(target, event):
            self.calib.s_right = event.new
            self.plot_sensor()
            self.pn_fig.param.trigger('object')

        def callback_msb(target, event):
            self.calib.s_bottom = event.new
            self.plot_sensor()
            self.pn_fig.param.trigger('object')

        def callback_msl(target, event):
            self.calib.s_left = event.new
            self.plot_sensor()
            self.pn_fig.param.trigger('object')

        def callback_smin(target, event):
            self.calib.s_min = event.new
            self.plot_sensor()
            self.pn_fig.param.trigger('object')

        def callback_smax(target, event):
            self.calib.s_max = event.new
            self.plot_sensor()
            self.pn_fig.param.trigger('object')

        s_margin_top = pn.widgets.IntSlider(name='Sensor top margin', value=self.calib.s_top, start=0,
                                            end=self.calib.s_height)
        s_margin_top.link(self.plot_sensor, callbacks={'value': callback_mst})

        s_margin_right = pn.widgets.IntSlider(name='Sensor right margin', value=self.calib.s_right, start=0,
                                              end=self.calib.s_width)
        s_margin_right.link(self.plot_sensor, callbacks={'value': callback_msr})

        s_margin_bottom = pn.widgets.IntSlider(name='Sensor bottom margin', value=self.calib.s_bottom, start=0,
                                               end=self.calib.s_height)
        s_margin_bottom.link(self.plot_sensor, callbacks={'value': callback_msb})

        s_margin_left = pn.widgets.IntSlider(name='Sensor left margin', value=self.calib.s_left, start=0,
                                             end=self.calib.s_width)
        s_margin_left.link(self.plot_sensor, callbacks={'value': callback_msl})

        s_min = pn.widgets.IntSlider(name='Sensor minimum', value=self.calib.s_min, start=0,
                                             end=2000)
        s_min.link(self.plot_sensor, callbacks={'value': callback_smin})

        s_max = pn.widgets.IntSlider(name='Sensor maximum', value=self.calib.s_max, start=0,
                                             end=2000)
        s_max.link(self.plot_sensor, callbacks={'value': callback_smax})

        widgets = pn.Column(s_margin_top, s_margin_right, s_margin_bottom, s_margin_left)
        panel = pn.Row(self.pn_fig, widgets)
        return panel


    def plot_sensor(self):
        # clear old axes
        self.ax.cla()

        rec_t = plt.Rectangle((0, self.calib.s_height - self.calib.s_top), self.calib.s_width,
                              self.calib.s_top, fc='r', alpha=0.3)
        rec_r = plt.Rectangle((self.calib.s_width - self.calib.s_right, 0), self.calib.s_right,
                              self.calib.s_height, fc='r', alpha=0.3)
        rec_b = plt.Rectangle((0, 0), self.calib.s_width, self.calib.s_bottom, fc='r', alpha=0.3)
        rec_l = plt.Rectangle((0, 0), self.calib.s_left, self.calib.s_height, fc='r', alpha=0.3)

        self.ax.imshow(self.frame, origin='lower')
        self.ax.add_patch(rec_t)
        self.ax.add_patch(rec_r)
        self.ax.add_patch(rec_b)
        self.ax.add_patch(rec_l)

        self.ax.set_axis_off()

        return True


class CalibrationOLD:
    """
    TODO:refactor completely! Make clear distinction between the calibration methods and calibration Data!
    Tune calibration parameters. Save calibration file. Have methods to project so we can see what we are calibrating
    """

    def __init__(self, associated_projector=None, associated_kinect=None, calibration_file=None,
                 json_calibration_file=None):
        """

        Args:
            associated_projector:
            associated_kinect:
            calibration_file:
        """
        self.id = 0
        self.associated_projector = associated_projector
        self.projector_resolution = associated_projector.resolution
        self.associated_kinect = associated_kinect
        if calibration_file is None:
            self.calibration_file = "calibration" + str(self.id) + ".dat"

        self.calibration_data = CalibrationData(
            legend_x_lim=(self.projector_resolution[1] - 50, self.projector_resolution[0] - 1),
            legend_y_lim=(self.projector_resolution[1] - 100, self.projector_resolution[1] - 50),
            profile_area=False,
            profile_x_lim=(self.projector_resolution[0] - 50, self.projector_resolution[0] - 1),
            profile_y_lim=(self.projector_resolution[1] - 100, self.projector_resolution[1] - 1),
            hot_area=False,
            hot_x_lim=(self.projector_resolution[0] - 50, self.projector_resolution[0] - 1),
            hot_y_lim=(self.projector_resolution[1] - 100, self.projector_resolution[1] - 1)
        )
        # new simplified json approach
        if json_calibration_file is not None:
            self.load_json(json_calibration_file)

        self.cmap = None
        self.contours = True
        self.n_contours = 20
        self.contour_levels = numpy.arange(self.calibration_data.z_range[0],
                                           self.calibration_data.z_range[1],
                                           float(self.calibration_data.z_range[1] - self.calibration_data.z_range[
                                               0]) / self.n_contours)

    # ...

    def load_json(self, file):
        with open(file) as calibration_json:
            self.calibration_data.__dict__ = json.load(calibration_json)
        print("JSON configuration loaded.")

    def save_json(self, file='calibration.json'):
        with open(file, "w") as calibration_json:
            json.dump(self.calibration_data.__dict__, calibration_json)
        print('JSON configuration file saved:', str(file))

    def load(self, calibration_file=None):
        """

        Args:
            calibration_file:

        Returns:

        """
        if calibration_file == None:
            calibration_file = self.calibration_file
        try:
            self.calibration_data = pickle.load(open(calibration_file, 'rb'))
            if not isinstance(self.calibration_data, CalibrationData):
                raise TypeError("loaded data is not a Calibration File object")
        except OSError:
            print("calibration data file not found. Using default values")

    def save(self, calibration_file=None):
        """

        Args:
            calibration_file:

        Returns:

        """
        if calibration_file is None:
            calibration_file = self.calibration_file
        pickle.dump(self.calibration_data, open(calibration_file, 'wb'))
        print("calibration saved to " + str(calibration_file))

    def create(self):
        """

        Returns:

        """
        if self.associated_projector is None:
            print("Error: no Projector instance found.")

        if self.associated_kinect is None:
            print("Error: no kinect instance found.")

        def calibrate(rot_angle, x_lim, y_lim, x_pos, y_pos, scale_factor, z_range, box_width, box_height, legend_area,
                      legend_x_lim, legend_y_lim, profile_area, profile_x_lim, profile_y_lim, hot_area, hot_x_lim,
                      hot_y_lim, close_click):
            """

            Args:
                rot_angle:
                x_lim:
                y_lim:
                x_pos:
                y_pos:
                scale_factor:
                z_range:
                box_width:
                box_height:
                legend_area:
                legend_x_lim:
                legend_y_lim:
                profile_area:
                profile_x_lim:
                profile_y_lim:
                hot_area:
                hot_x_lim:
                hot_y_lim:
                close_click:

            Returns:

            """
            depth = self.associated_kinect.get_frame()
            depth_rotated = scipy.ndimage.rotate(depth, rot_angle, reshape=False)
            depth_cropped = depth_rotated[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            depth_masked = numpy.ma.masked_outside(depth_cropped, self.calibration_data.z_range[0],
                                                   self.calibration_data.z_range[
                                                       1])  # depth pixels outside of range are white, no data pixe;ls are black.

            self.cmap = matplotlib.colors.Colormap('viridis')
            self.cmap.set_bad('white', 800)
            plt.set_cmap(self.cmap)
            h = (y_lim[1] - y_lim[0]) / 100.0
            w = (x_lim[1] - x_lim[0]) / 100.0

            fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.pcolormesh(depth_masked, vmin=self.calibration_data.z_range[0],
                          vmax=self.calibration_data.z_range[1])

            if self.contours is True:  # draw contours
                self.contour_levels = numpy.arange(self.calibration_data.z_range[0],
                                                   self.calibration_data.z_range[1],
                                                   float(self.calibration_data.z_range[1] -
                                                         self.calibration_data.z_range[
                                                             0]) / self.n_contours)  # update contour levels
                plt.contour(depth_masked, levels=self.contour_levels, linewidths=1.0, colors=[(0, 0, 0, 1.0)])

            plt.savefig(os.path.join(self.associated_projector.work_directory, 'current_frame.png'), pad_inches=0)
            plt.close(fig)

            self.calibration_data = CalibrationData(
                rot_angle=rot_angle,
                x_lim=x_lim,
                y_lim=y_lim,
                x_pos=x_pos,
                y_pos=y_pos,
                scale_factor=scale_factor,
                z_range=z_range,
                box_width=box_width,
                box_height=box_height,
                legend_area=legend_area,
                legend_x_lim=legend_x_lim,
                legend_y_lim=legend_y_lim,
                profile_area=profile_area,
                profile_x_lim=profile_x_lim,
                profile_y_lim=profile_y_lim,
                hot_area=hot_area,
                hot_x_lim=hot_x_lim,
                hot_y_lim=hot_y_lim
            )

            if self.calibration_data.legend_area is not False:
                legend = Image.new('RGB', (
                    self.calibration_data.legend_x_lim[1] - self.calibration_data.legend_x_lim[0],
                    self.calibration_data.legend_y_lim[1] - self.calibration_data.legend_y_lim[0]), color='white')
                ImageDraw.Draw(legend).text((10, 10), "Legend", fill=(255, 255, 0))
                legend.save(os.path.join(self.associated_projector.work_directory, 'legend.png'))
            if self.calibration_data.profile_area is not False:
                profile = Image.new('RGB', (
                    self.calibration_data.profile_x_lim[1] - self.calibration_data.profile_x_lim[0],
                    self.calibration_data.profile_y_lim[1] - self.calibration_data.profile_y_lim[0]), color='blue')
                ImageDraw.Draw(profile).text((10, 10), "Profile", fill=(255, 255, 0))
                profile.save(os.path.join(self.associated_projector.work_directory, 'profile.png'))
            if self.calibration_data.hot_area is not False:
                hot = Image.new('RGB', (self.calibration_data.hot_x_lim[1] - self.calibration_data.hot_x_lim[0],
                                        self.calibration_data.hot_y_lim[1] - self.calibration_data.hot_y_lim[0]),
                                color='red')
                ImageDraw.Draw(hot).text((10, 10), "Hot Area", fill=(255, 255, 0))
                hot.save(os.path.join(self.associated_projector.work_directory, 'hot.png'))
            self.associated_projector.show()
            if close_click == True:
                calibration_widget.close()

        calibration_widget = widgets.interactive(calibrate,
                                                 rot_angle=widgets.IntSlider(
                                                     value=self.calibration_data.rot_angle, min=-180, max=180,
                                                     step=1, continuous_update=False),
                                                 x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.x_lim[0],
                                                            self.calibration_data.x_lim[1]],
                                                     min=0, max=640, step=1, continuous_update=False),
                                                 y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.y_lim[0],
                                                            self.calibration_data.y_lim[1]],
                                                     min=0, max=480, step=1, continuous_update=False),
                                                 x_pos=widgets.IntSlider(value=self.calibration_data.x_pos, min=0,
                                                                         max=self.projector_resolution[0]),
                                                 y_pos=widgets.IntSlider(value=self.calibration_data.y_pos, min=0,
                                                                         max=self.projector_resolution[1]),
                                                 scale_factor=widgets.FloatSlider(
                                                     value=self.calibration_data.scale_factor, min=0.1, max=6.0,
                                                     step=0.01, continuous_update=False),
                                                 z_range=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.z_range[0],
                                                            self.calibration_data.z_range[1]],
                                                     min=500, max=2000, step=1, continuous_update=False),
                                                 box_width=widgets.IntSlider(value=self.calibration_data.box_dim[0],
                                                                             min=0,
                                                                             max=2000, continuous_update=False),
                                                 box_height=widgets.IntSlider(value=self.calibration_data.box_dim[1],
                                                                              min=0,
                                                                              max=2000, continuous_update=False),
                                                 legend_area=widgets.ToggleButton(
                                                     value=self.calibration_data.legend_area,
                                                     description='display a legend',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Description',
                                                     icon='check'),
                                                 legend_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.legend_x_lim[0],
                                                            self.calibration_data.legend_x_lim[1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 legend_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.legend_y_lim[0],
                                                            self.calibration_data.legend_y_lim[1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 profile_area=widgets.ToggleButton(
                                                     value=self.calibration_data.profile_area,
                                                     description='display a profile area',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='display a profile area',
                                                     icon='check'),
                                                 profile_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.profile_x_lim[0],
                                                            self.calibration_data.profile_x_lim[1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 profile_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.profile_y_lim[0],
                                                            self.calibration_data.profile_y_lim[1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 hot_area=widgets.ToggleButton(
                                                     value=self.calibration_data.hot_area,
                                                     description='display a hot area for qr codes',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='display a hot area for qr codes',
                                                     icon='check'),
                                                 hot_x_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.hot_x_lim[0],
                                                            self.calibration_data.hot_x_lim[1]],
                                                     min=0, max=self.projector_resolution[0], step=1,
                                                     continuous_update=False),
                                                 hot_y_lim=widgets.IntRangeSlider(
                                                     value=[self.calibration_data.hot_y_lim[0],
                                                            self.calibration_data.hot_y_lim[1]],
                                                     min=0, max=self.projector_resolution[1], step=1,
                                                     continuous_update=False),
                                                 close_click=widgets.ToggleButton(
                                                     value=False,
                                                     description='Close calibration',
                                                     disabled=False,
                                                     button_style='',  # 'success', 'info', 'warning', 'danger' or ''
                                                     tooltip='Close calibration',
                                                     icon='check'
                                                 )

                                                 )
        IPython.display.display(calibration_widget)


class Projector:

    dpi = 100 # make sure that figures can be displayed pixel-precise

    def __init__(self, calibrationdata):
        self.calib = calibrationdata
        #self.plot = plot

        # panel components (panes)
        self.frame = None
        self.legend = None
        self.panel = None

        self.create_panel() # make explicit?

    def show(self, figure):
        # TODO: Fix that nasty memory leak!
        self.frame.object = figure
        #plt.close()

    def create_panel(self):

        css = '''
        body {
          margin:0px;
          background-color: #ffffff;
        }
        .bk.frame {
        }
        .bk.legend {
          background-color: #AAAAAA;
        }
        .panel {
          background-color: #000000;
        }
        '''

        pn.extension(raw_css=[css])
        # Create a panel object and serve it within an external bokeh browser that will be opened in a separate window
        # in this special case, a "tight" layout would actually add again white space to the plt canvas, which was already cropped by specifying limits to the axis


        self.frame = pn.pane.Matplotlib(plt.figure(), width=self.calib.p_frame_width, height=self.calib.p_frame_height,
                                         margin=[self.calib.p_frame_top, 0, 0, self.calib.p_frame_left], tight=False, dpi=self.dpi, css_classes=['frame'])

        self.legend = pn.Column("<br>\n# Legend",
                                margin=[self.calib.p_frame_top, 0, 0, 0],
                                css_classes=['legend'])

        # Combine panel and deploy bokeh server
        self.panel = pn.Row(self.frame, self.legend, width=self.calib.p_width, height=self.calib.p_height,
                            sizing_mode='fixed', css_classes=['panel'])

        # TODO: Add specific port? port=4242
        self.panel.show(threaded=False)

    def trigger(self):
        self.frame.param.trigger('object')

    def calibrate_projector(self):

        margin_top = pn.widgets.IntSlider(name='Top margin', value=self.calib.p_frame_top, start=0, end=200)
        def callback_mt(target, event):
            m = target.margin
            n = event.new
            # just changing single indices does not trigger updating of pane
            target.margin = [n, m[1], m[2], m[3]]
            # also update calibration object
            self.calib.p_frame_top = event.new

        margin_top.link(self.frame, callbacks={'value': callback_mt})

        margin_left = pn.widgets.IntSlider(name='Left margin', value=self.calib.p_frame_left, start=0, end=200)
        def callback_ml(target, event):
            m = target.margin
            n = event.new
            # just changing single indices does not trigger updating of pane
            target.margin = [m[0], m[1], m[2], n]
            # also update calibration object
            self.calib.p_frame_left = event.new
        margin_left.link(self.frame, callbacks={'value': callback_ml})

        width = pn.widgets.IntSlider(name='Map width', value=self.calib.p_frame_width, start=self.calib.p_frame_width - 400, end=self.calib.p_frame_width + 400)
        def callback_width(target, event):
            target.width = event.new
            target.param.trigger('object')
            # also update calibration object
            self.calib.p_frame_width = event.new
        width.link(self.frame, callbacks={'value': callback_width})

        height = pn.widgets.IntSlider(name='Map height', value=self.calib.p_frame_height, start=self.calib.p_frame_height - 400, end=self.calib.p_frame_height + 400)
        def callback_height(target, event):
            target.height = event.new
            target.param.trigger('object')
            # also update calibration object
            self.calib.p_frame_height = event.new
            #self.plot.redraw_plot()
        height.link(self.frame, callbacks={'value': callback_height})

        widgets = pn.Column("### Map positioning", margin_top, margin_left, width, height)
        return widgets


class CalibrationData:
    """

    """

    def __init__(self,
                 p_width=800, p_height=600, p_frame_top=0, p_frame_left=0,
                 p_frame_width=600, p_frame_height=450,
                 s_top=0, s_right=0, s_bottom=0, s_left=0, s_min=700, s_max=1500,
                 file=None):
        """

        Args:
            p_width=800
            p_height=600
            p_frame_top=0
            p_frame_left=0
            p_frame_width=600
            p_frame_height=450
            s_top=0
            s_right=0
            s_bottom=0
            s_left=0
            s_min=700
            s_max=1500
            file=None

        Returns:
            None

        """

        # version identifier (will be changed if new calibration parameters are introduced / removed)
        self.version = "0.8alpha"

        # projector
        self.p_width = p_width
        self.p_height = p_height

        self.p_frame_top = p_frame_top
        self.p_frame_left = p_frame_left
        self.p_frame_width = p_frame_width
        self.p_frame_height = p_frame_height

        #self.p_legend_top =
        #self.p_legend_left =
        #self.p_legend_width =
        #self.p_legend_height =

        # hot area
        #self.p_hot_top =
        #self.p_hot_left =
        #self.p_hot_width =
        #self.p_hot_height =

        # profile area
        #self.p_profile_top =
        #self.p_profile_left =
        #self.p_profile_width =
        #self.p_profile_height =

        # sensor (e.g. Kinect)
        self.s_name = 'generic' # name to identify the associated sensor device
        self.s_width = 500 # will be updated by sensor init
        self.s_height = 400 # will be updated by sensor init

        self.s_top = s_top
        self.s_right = s_right
        self.s_bottom = s_bottom
        self.s_left = s_left
        self.s_min = s_min
        self.s_max = s_max

        if file is not None:
            self.load_json(file)

    # computed parameters for easy access
    @property
    def s_frame_width(self):
        return self.s_width - self.s_left - self.s_right

    @property
    def s_frame_height(self):
        return self.s_height - self.s_top - self.s_bottom

    @property
    def scale_factor(self):
        return (self.p_frame_width / self.s_frame_width), (self.p_frame_height / self.s_frame_height)

    # JSON import/export
    def load_json(self, file):
        with open(file) as calibration_json:
            data = json.load(calibration_json)
            if data['version'] == self.version:
                self.__dict__ = data
                print("JSON configuration loaded.")
            else:
                print("JSON configuration incompatible.\nPlease recalibrate manually!")

    def save_json(self, file='calibration.json'):
        with open(file, "w") as calibration_json:
            json.dump(self.__dict__, calibration_json)
        print('JSON configuration file saved:', str(file))

class Scale:
    """
    class that handles the scaling of whatever the sandbox shows and the real world sandbox
    self.extent: 3d extent of the model in the sandbox in model units.

    """

    def __init__(self, calibration: Calibration = None, xy_isometric=True, extent=None):
        """

        Args:
            calibration:
            xy_isometric:
            extent:
        """
        self.calibration = calibration
        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        """
        self.xy_isometric = xy_isometric
        self.scale = [None, None, None]
        self.pixel_size = [None, None]
        self.pixel_scale = [None, None]
        self.output_res = None

        if extent is None:  # extent should be array with shape (6,) or convert to list?
            self.extent = numpy.asarray([
                0.0,
                self.calibration.calibration_data.box_width,
                0.0,
                self.calibration.calibration_data.box_height,
                self.calibration.calibration_data.z_range[0],
                self.calibration.calibration_data.z_range[1],
            ])

        else:
            self.extent = numpy.asarray(extent)  # check: array with 6 entries!

    def calculate_scales(self):
        """
        calculates the factors for the coordinates transformation kinect-extent

        Returns:
            nothing, but changes in place:
            self.output_res [pixels]: width and height of sandbox image
            self.pixel_scale [modelunits/pixel]: XY scaling factor
            pixel_size [mm/pixel]
            self.scale

        """

        self.output_res = (self.calibration.calibration_data.x_lim[1] -
                           self.calibration.calibration_data.x_lim[0],
                           self.calibration.calibration_data.y_lim[1] -
                           self.calibration.calibration_data.y_lim[0])
        self.pixel_scale[0] = float(self.extent[1] - self.extent[0]) / float(self.output_res[0])
        self.pixel_scale[1] = float(self.extent[3] - self.extent[2]) / float(self.output_res[1])
        self.pixel_size[0] = float(self.calibration.calibration_data.box_width) / float(self.output_res[0])
        self.pixel_size[1] = float(self.calibration.calibration_data.box_height) / float(self.output_res[1])

        # TODO: change the extrent in place!! or create a new extent object that stores the extent after that modification.
        if self.xy_isometric == True:  # model is extended in one horizontal direction to fit  into box while the scale
            # in both directions is maintained
            print("Aspect ratio of the model is fixed in XY")
            if self.pixel_scale[0] >= self.pixel_scale[1]:
                self.pixel_scale[1] = self.pixel_scale[0]
                print("Model size is limited by X dimension")
            else:
                self.pixel_scale[0] = self.pixel_scale[1]
                print("Model size is limited by Y dimension")

        self.scale[0] = self.pixel_scale[0] / self.pixel_size[0]
        self.scale[1] = self.pixel_scale[1] / self.pixel_size[1]
        self.scale[2] = float(self.extent[5] - self.extent[4]) / (
                self.calibration.calibration_data.z_range[1] -
                self.calibration.calibration_data.z_range[0])
        print("scale in Model units/ mm (X,Y,Z): " + str(self.scale))

    # TODO: manually define zscale and either lower or upper limit of Z, adjust rest accordingly.


class Grid:
    """
    class for grid objects. a grid stores the 3D coordinate of each pixel recorded by the kinect in model coordinates
    a calibration object must be provided, it is used to crop the kinect data to the area of interest
    TODO:  The cropping should be done in the kinect class, with calibration_data passed explicitly to the method! Do this for all the cases where calibration data is needed!
    """

    def __init__(self, calibration=None, scale=None, ):
        """

        Args:
            calibration:
            scale:

        Returns:
            None

        """


        self.calibration = calibration
        """
        if isinstance(calibration, Calibration):
            self.calibration = calibration
        else:
            raise TypeError("you must pass a valid calibration instance")
        """
        if isinstance(scale, Scale):
            self.scale = scale
        else:
            self.scale = Scale(calibration=self.calibration)
            print("no scale provided or scale invalid. A default scale instance is used")
        self.empty_depth_grid = None
        self.create_empty_depth_grid()

    def create_empty_depth_grid(self):
        """
        Sets up XY grid (Z is empty that is the name coming from)

        Returns:

        """

        grid_list = []
        self.output_res = (self.calibration.calibration_data.x_lim[1] -
                           self.calibration.calibration_data.x_lim[0],
                           self.calibration.calibration_data.y_lim[1] -
                           self.calibration.calibration_data.y_lim[0])
        """compare:
        for x in range(self.output_res[1]):
            for y in range(self.output_res[0]):
                grid_list.append([y * self.scale.pixel_scale[1] + self.scale.extent[2], x * self.scale.pixel_scale[0] + self.scale.extent[0]])
        """

        for y in range(self.output_res[1]):
            for x in range(self.output_res[0]):
                grid_list.append([x * self.scale.pixel_scale[0] + self.scale.extent[0],
                                  y * self.scale.pixel_scale[1] + self.scale.extent[2]])

        empty_depth_grid = numpy.array(grid_list)
        self.empty_depth_grid = empty_depth_grid
        self.depth_grid = None  # I know, this should have thew right type.. anyway.
        print("the shown extent is [" + str(self.empty_depth_grid[0, 0]) + ", " +
              str(self.empty_depth_grid[-1, 0]) + ", " +
              str(self.empty_depth_grid[0, 1]) + ", " +
              str(self.empty_depth_grid[-1, 1]) + "] "
              )

        # return self.empty_depth_grid

    def update_grid(self, depth):
        """
        Appends the z (depth) coordinate to the empty depth grid.
        this has to be done every frame while the xy coordinates only change if the calibration or model extent is changed.
        For performance reasons these steps are therefore separated.

        Args:
            depth:

        Returns:

        """

        # TODO: is this flip still necessary?
        depth = numpy.fliplr(depth)  ##dirty workaround to get the it running with new gempy version.
        filtered_depth = numpy.ma.masked_outside(depth, self.calibration.calibration_data.z_range[0],
                                                 self.calibration.calibration_data.z_range[1])
        scaled_depth = self.scale.extent[5] - (
                (filtered_depth - self.calibration.calibration_data.z_range[0]) / (
                self.calibration.calibration_data.z_range[1] -
                self.calibration.calibration_data.z_range[0]) * (self.scale.extent[5] - self.scale.extent[4]))
        rotated_depth = scipy.ndimage.rotate(scaled_depth, self.calibration.calibration_data.rot_angle,
                                             reshape=False)
        cropped_depth = rotated_depth[self.calibration.calibration_data.y_lim[0]:
                                      self.calibration.calibration_data.y_lim[1],
                        self.calibration.calibration_data.x_lim[0]:
                        self.calibration.calibration_data.x_lim[1]]

        flattened_depth = numpy.reshape(cropped_depth, (numpy.shape(self.empty_depth_grid)[0], 1))
        depth_grid = numpy.concatenate((self.empty_depth_grid, flattened_depth), axis=1)

        self.depth_grid = depth_grid


class Contour:  # TODO: change the whole thing to use keyword arguments!!
    """
    class to handle contour lines in the sandbox. contours can shpow depth or anything else.
    TODO: pass on keyword arguments to the plot and label functions for more flexibility

    """

    def __init__(self, start, end, step, show=True, show_labels=False, linewidth=1.0, colors=[(0, 0, 0, 1.0)],
                 inline=0, fontsize=15, label_format='%3.0f'):
        """

        Args:
            start:
            end:
            step:
            show:
            show_labels:
            linewidth:
            colors:
            inline:
            fontsize:
            label_format:

        Returns:
            None

        """
        self.start = start
        self.end = end
        self.step = step
        self.show = show
        self.show_labels = show_labels
        self.linewidth = linewidth
        self.colors = colors
        self.levels = numpy.arange(self.start, self.end, self.step)
        self.contours = None
        self.data = None  # Data has to be updated for each frame

        # label attributes:
        self.inline = inline
        self.fontsize = fontsize
        self.label_format = label_format


class Plot:
    """
    handles the plotting of a sandbox model

    """

    dpi = 100 # make sure that figures can be displayed pixel-precise

    def __init__(self, calibrationdata, contours=True, cmap=None, norm=None, lot=None):

        self.calib = calibrationdata

        self.cmap = cmap
        self.norm = norm
        self.lot = lot

        # flags
        self.contours = contours
        #self.points = True

        self.figure = None
        self.ax = None # current plot composition

        self.create_empty_frame() # initial figure for starting projector


    # def __init__(self, calibration=None, cmap=None, norm=None, lot=None, outfile=None):

    #     if isinstance(calibration, Calibration):
    #         self.calibration = calibration
    #     else:
    #         raise TypeError("you must pass a valid calibration instance")

    #     self.output_res = (
    #         self.calibration.calibration_data.x_lim[1] -
    #         self.calibration.calibration_data.x_lim[0],
    #         self.calibration.calibration_data.y_lim[1] -
    #         self.calibration.calibration_data.y_lim[0]
    #     )
    #
    #     self.h = self.calibration.calibration_data.scale_factor * (self.output_res[1]) / 100.0
    #     self.w = self.calibration.calibration_data.scale_factor * (self.output_res[0]) / 100.0

    def create_empty_frame(self):
        self.figure = plt.figure(figsize=(self.calib.p_frame_width / self.dpi,
                                          self.calib.p_frame_height / self.dpi),
                                 dpi=self.dpi)  # , frameon=False) # curent figure
        self.ax = plt.Axes(self.figure, [0., 0., 1., 1.])
        self.figure.add_axes(self.ax)

        self.ax.set_axis_off()

    def render_frame(self, data):
        # clear axes to draw new ones on figure
        self.ax.cla()

        #self.block = rasterdata.reshape((self.calib.s_frame_height, self.calib.s_frame_width))
        #self.ax.pcolormesh(self.block,
        self.ax.pcolormesh(data, cmap=self.cmap, norm=self.norm)
        if self.contours:
            self.ax.contour(data, colors='k')


        # crop axis (!!!input dimensions of calibrated sensor!!!)
        self.ax.axis([0, self.calib.s_frame_width, 0, self.calib.s_frame_height])
        self.ax.set_axis_off()

        # return final figure
        #return self.figure
        return True

    def add_contours(self, contour, data): # TODO: Check compability
        """
        renders contours to the current plot object. \
        The data has to come in a specific shape as needed by the matplotlib contour function.
        we explicity enforce to provide X and Y at this stage (you are welcome to change this)

        Args:
            contour: a contour instance
            data:  a list with the form x,y,z
                x: list of the coordinates in x direction (e.g. range(Scale.output_res[0])
                y: list of the coordinates in y direction (e.g. range(Scale.output_res[1])
                z: 2D array-like with the values to be contoured

        Returns:

        """

        if contour.show is True:
            contour.contours = self.ax.contour(data[0], data[1], data[2], levels=contour.levels,
                                               linewidths=contour.linewidth, colors=contour.colors)
            if contour.show_labels is True:
                self.ax.clabel(contour.contours, inline=contour.inline, fontsize=contour.fontsize,
                               fmt=contour.label_format)

    def add_lith_contours(self, block, levels=None):
        """

        Args:
            block:
            levels:

        Returns:

        """
        plt.contourf(block, levels=levels, cmap=self.cmap, norm=self.norm, extend="both")

    def create_legend(self):
        """ Returns:
        """
        pass

class Module:
    """
    Parent Module with threading methods and abstract attributes and methods for child classes
    """
    __metaclass__ = ABCMeta

    def __init__(self, calibrationdata, sensor, projector, crop = True, **kwargs):
        self.calib = calibrationdata
        self.sensor = sensor
        self.projector = projector
        self.plot = Plot(self.calib, **kwargs)

        # flags
        self.crop = crop

        # threading
        self._lock = threading.Lock()
        self.thread = None
        self.thread_running = False

    @abstractmethod
    def setup(self):
        # Wildcard: Everything necessary to set up before an model update can be performed.
        pass

    @abstractmethod
    def update(self):
        # Wildcard: Single model update operation that can be looped in a thread.
        pass

    def thread_loop(self):
        while self.thread_running:
            self.update()

    def run(self):
        if not self.thread_running:
            self.thread_running = True
            self.thread = threading.Thread(target=self.thread_loop, daemon=True, )
            self.thread.start()
            print('Thread started...')
        else:
            print('Thread already running. First stop with stop().')

    def stop(self):
        self.thread_running = False  # set flag to end thread loop
        self.thread.join()  # wait for the thread to finish
        print('Thread stopped.')

    def crop_frame(self, frame):
        return frame[self.calib.s_bottom:-self.calib.s_top, self.calib.s_left:-self.calib.s_right]


class TopoModule(Module):
    """
    Module for simple Topography visualization without computing a geological model
    """
    def setup(self):
        with self._lock:
            frame = self.sensor.get_filtered_frame()
            if self.crop is True:
                frame = self.crop_frame(frame)
            self.plot.render_frame(frame)
        self.projector.frame.object = self.plot.figure

    def update(self):
        with self._lock:
            frame = self.sensor.get_filtered_frame()
            if self.crop is True:
                frame = self.crop_frame(frame)

        self.plot.render_frame(frame)
        #self.projector.show(fig)
        self.projector.trigger()


class BlockModule(Module):
    # child class of Model

    def __init__(self):
        self.lithology = None
        self.faults = None
        self.fluid_contact = None


    def setup(self):
        pass

    def update(self):
        with self.lock:
            pass

    def parse_block_vip(self, target, infile):
        pass

    def rescale_block(self, interpolate=True):
        pass

class GemPyModule(Module):
    # child class of Model
    pass


class GeoMapModule:
    """

    """

    # TODO: When we move GeoMapModule import gempy just there

    def __init__(self, geo_model, grid: Grid, geol_map: Plot):
        """

        Args:
            geo_model:
            grid:
            geol_map:
            work_directory:

        Returns:
            None

        """

        self.geo_model = geo_model
        self.kinect_grid = grid
        self.geol_map = geol_map

        self.fault_line = self.create_fault_line(0, self.geo_model.geo_data_res.n_faults + 0.5001)
        self.main_contours = self.create_main_contours(self.kinect_grid.scale.extent[4],
                                                       self.kinect_grid.scale.extent[5])
        self.sub_contours = self.create_sub_contours(self.kinect_grid.scale.extent[4],
                                                     self.kinect_grid.scale.extent[5])

        self.x_grid = range(self.kinect_grid.scale.output_res[0])
        self.y_grid = range(self.kinect_grid.scale.output_res[1])

        self.plot_topography = True
        self.plot_faults = True

    def compute_model(self, kinect_array):
        """

        Args:
            kinect_array:

        Returns:

        """
        self.kinect_grid.update_grid(kinect_array)
        sol = gp.compute_model_at(self.kinect_grid.depth_grid, self.geo_model)
        lith_block = sol[0][0]
        fault_blocks = sol[1][0::2]
        block = lith_block.reshape((self.kinect_grid.scale.output_res[1],
                                    self.kinect_grid.scale.output_res[0]))

        return block, fault_blocks

    # TODO: Miguel: outfile folder should follow by default whatever is set in projection!
    # TODO: Temporal fix. Eventually we need a container class or metaclass with this data
    def render_geo_map(self, block, fault_blocks):
        """

        Args:
            block:
            fault_blocks:
            outfile:

        Returns:

        """

        self.geol_map.render_frame(block)

        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        # This line is for GemPy 1.2: fault_data = sol.fault_blocks.reshape((scalgeol_map.outfilee.output_res[1],
        # scale.output_res[0]))

        if self.plot_faults is True:
            for fault in fault_blocks:
                fault = fault.reshape((self.kinect_grid.scale.output_res[1], self.kinect_grid.scale.output_res[0]))
                self.geol_map.add_contours(self.fault_line, [self.x_grid, self.y_grid, fault])
        if self.plot_topography is True:
            self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
            self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])

        return self.geol_map.figure


    def create_fault_line(self,
                          start=0.5,
                          end=50.5,  # TODO Miguel:increase?
                          step=1.0,
                          linewidth=3.0,
                          colors=[(1.0, 1.0, 1.0, 1.0)]):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:

        Returns:

        """

        self.fault_line = Contour(start=start, end=end, step=step, linewidth=linewidth,
                                  colors=colors)

        return self.fault_line

    def create_main_contours(self, start, end, step=100, linewidth=1.0,
                             colors=[(0.0, 0.0, 0.0, 1.0)], show_labels=True):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:
            show_labels:

        Returns:

        """

        self.main_contours = Contour(start=start,
                                     end=end,
                                     step=step,
                                     show_labels=show_labels,
                                     linewidth=linewidth, colors=colors)
        return self.main_contours

    def create_sub_contours(self,
                            start,
                            end,
                            step=25,
                            linewidth=0.8,
                            colors=[(0.0, 0.0, 0.0, 0.8)],
                            show_labels=False
                            ):
        """

        Args:
            start:
            end:
            step:
            linewidth:
            colors:
            show_labels:

        Returns:

        """

        self.sub_contours = Contour(start=start, end=end, step=step, linewidth=linewidth, colors=colors,
                                    show_labels=show_labels)
        return self.sub_contours

    def export_topographic_map(self, output="topographic_map.pdf"):
        """

        Args:
            output:

        Returns:

        """
        self.geol_map.create_empty_frame()
        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
        self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])
        self.geol_map.save(outfile=output)

    def export_geological_map(self, kinect_array, output="geological_map.pdf"):
        """

        Args:
            kinect_array:
            output:

        Returns:

        """

        print("there is still a bug in the map that causes the uppermost lithology to be displayed in the basement"
              " color. Unfortunately we do not have a quick fix for this currently... Sorry! Please fix the map "
              "yourself, for example using illustrator")

        lith_block, fault_blocks = self.compute_model(kinect_array)

        # This line is for GemPy 1.2: lith_block = sol.lith_block.reshape((scale.output_res[1], scale.output_res[0]))

        self.geol_map.create_empty_frame()

        lith_levels = self.geo_model.potential_at_interfaces[-1].sort()
        self.geol_map.add_lith_contours(lith_block, levels=lith_levels)

        elevation = self.kinect_grid.depth_grid.reshape((self.kinect_grid.scale.output_res[1],
                                                         self.kinect_grid.scale.output_res[0], 3))[:, :, 2]
        # This line is for GemPy 1.2: fault_data = sol.fault_blocks.reshape((scalgeol_map.outfilee.output_res[1],
        # scale.output_res[0]))

        if self.plot_faults is True:
            for fault in fault_blocks:
                fault = fault.reshape((self.kinect_grid.scale.output_res[1], self.kinect_grid.scale.output_res[0]))
                self.geol_map.add_contours(self.fault_line, [self.x_grid, self.y_grid, fault])

        if self.plot_topography is True:
            self.geol_map.add_contours(self.main_contours, [self.x_grid, self.y_grid, elevation])
            self.geol_map.add_contours(self.sub_contours, [self.x_grid, self.y_grid, elevation])

        self.geol_map.save(outfile=output)


class SandboxThread:
    """
    container for modules that handles threading. any kind of module can be loaded, as long as it contains a 'setup' and 'render_frame" method!
    """

    def __init__(self, module, kinect, projector, path=None):
        """

        Args:
            module:
            kinect:
            projector:
            path:
        """
        self.module = module
        self.kinect = kinect
        self.projector = projector
        self.path = path
        self.thread = None
        self.lock = threading.Lock()
        self.stop_thread = False
        self.plot = None

    def loop(self):
        """

        Returns:

        """
        while self.stop_thread is False:
            depth = self.kinect.get_filtered_frame()
            with self.lock:
                # TODO: Making the next two lines agnostic from GemPy
                lith, fault = self.module.compute_model(depth)
                self.plot=self.module.render_geo_map(lith, fault)

                self.projector.show(self.plot)

    def run(self):
        """

        Returns:

        """
        self.stop_thread = False
        self.thread = threading.Thread(target=self.loop, daemon=None)
        self.thread.start()
        # with thread and thread lock move these to main sandbox

    def pause(self):
        """

        Returns:

        """
        self.lock.release()

    def resume(self):
        """

        Returns:

        """
        self.lock.acquire()

    def kill(self):
        """

        Returns:

        """
        self.stop_thread = True
        try:
            self.lock.release()
        except:
            pass


class ArucoMarkers:
    """
    class to detect Aruco markers in the kinect data (IR and RGB)
    An Area of interest can be specified, markers outside this area will be ignored
    TODO: run as loop in a thread, probably implement in API
    """

    def __init__(self, aruco_dict=None, Area=None):
        if not aruco_dict:
            self.aruco_dict = aruco.DICT_4X4_50  # set the default dictionary here
        else:
            self.aruco_dict = aruco_dict
        self.Area = Area  # set a square Area of interest here (Hot-Area)
        self.kinect = KinectV2()
        self.ir_markers = self.find_markers_ir(self.kinect)
        self.rgb_markers = self.find_markers_rgb(self.kinect)
        self.dict_markers_current = self.update_dict_markers_current()  # markers that were detected in the last frame
        #self.dict_markers_all =pd.DataFrame({}) # all markers ever detected with their last known position and timestamp
        self.dict_markers_all = self.dict_markers_current
        self.lock = threading.Lock  # thread lock object to avoid read-write collisions in multithreading.
        #self.trs_dst = self.change_point_RGB_to_DepthIR()
        self.ArucoImage = self.create_aruco_marker()


    def get_location_marker(self, corners):
        pr1 = int(numpy.mean(corners[:, 0]))
        pr2 = int(numpy.mean(corners[:, 1]))
        return pr1, pr2

    def aruco_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(self.aruco_dict)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        return corners, ids, rejectedImgPoints

    def find_markers_ir(self, kinect: KinectV2):
        labels = {'ids', 'Corners_IR_x', 'Corners_IR_y'} #TODO: add orientation of aruco marker
        df = pd.DataFrame(columns=labels)
        list_values = df.set_index('ids')

        while len(list_values) < 4:

            minim = 0
            maxim = numpy.arange(1000, 30000, 500)
            IR = kinect.get_ir_frame_raw()
            for i in maxim:
                ir_use = numpy.interp(IR, (minim, i), (0, 255)).astype('uint8')
                ir3 = numpy.stack((ir_use, ir_use, ir_use), axis=2)
                corners, ids, rejectedImgPoints = self.aruco_detect(ir3)

                if not ids is None:
                    for j in range(len(ids)):
                        if ids[j] not in list_values.index.values:
                            x_loc, y_loc = self.get_location_marker(corners[j][0])
                            df_temp = pd.DataFrame({'ids': [ids[j][0]], 'Corners_IR_x': [x_loc], 'Corners_IR_y': [y_loc]})
                            df = pd.concat([df, df_temp], sort=False)
                            list_values = df.set_index('ids')

        self.ir_markers = list_values

        return self.ir_markers

    def find_markers_rgb(self, kinect :KinectV2):
        labels = {"ids", "Corners_RGB_x", "Corners_RGB_y"}  #TODO: add orientation of aruco marker
        df = pd.DataFrame(columns=labels)
        list_values_color = df.set_index("ids")

        while len(list_values_color) < 5:
            color = kinect.get_color()
            corners, ids, rejectedImgPoints = self.aruco_detect(color)

            if not ids is None:
                for j in range(len(ids)):
                    if ids[j] not in list_values_color.index.values:
                        x_loc, y_loc = self.get_location_marker(corners[j][0])
                        df_temp = pd.DataFrame({"ids": [ids[j][0]], "Corners_RGB_x": [x_loc], "Corners_RGB_y": [y_loc]})
                        df = pd.concat([df, df_temp], sort=False)
                        list_values_color = df.set_index("ids")

        self.rgb_markers = list_values_color

        return self.rgb_markers


    def update_dict_markers_current(self):

        ir_aruco_locations = self.ir_markers
        rgb_aruco_locations = self.rgb_markers
        self.dict_markers_current = pd.concat([ir_aruco_locations,rgb_aruco_locations], axis=1)
        return self.dict_markers_current

    def update_dict_markers_all(self):

        self.dict_markers_all.update(self.dict_markers_current)
        return self.dict_markers_all


    def erase_dict_markers_all(self):
        self.dict_markers_all = pd.DataFrame({})
        return self.dict_markers_all

    def change_point_RGB_to_DepthIR(self):
        """
        Get a perspective transform matrix to project points from RGB to Depth/IR space

        Args:
            src: location in x and y of the points from the source image (requires 4 points)
            dst: equivalence of position x and y from source image to destination image (requires 4 points)

        Returns:
            trs_dst: location in x and y of the projected point in Depth/IR space
        """
        full = self.dict_markers_current.dropna()
        mis = self.dict_markers_current[self.dict_markers_current.isna().any(1)]

        src = numpy.array(full[["Corners_RGB_x", "Corners_RGB_y"]]).astype(numpy.float32)
        dst = numpy.array(full[["Corners_IR_x", "Corners_IR_y"]]).astype(numpy.float32)

        trs_src = numpy.array([mis["Corners_RGB_x"], mis["Corners_RGB_y"], 1]).astype(numpy.float32)

        transform_perspective = cv2.getPerspectiveTransform(src, dst)

        trans_val = numpy.dot(transform_perspective, trs_src.T).astype("int")

        values = {"Corners_IR_x": trans_val[0], "Corners_IR_y": trans_val[1]}

        self.dict_markers_current = self.dict_markers_current.fillna(value=values)

        return self.dict_markers_current

    def create_aruco_marker(self, nx=1, ny=1,show=False):
        self.ArucoImage = 0
        if show is True:
            aruco_dictionary = aruco.Dictionary_get(self.aruco_dict)

            fig = plt.figure()
            for i in range(1, nx * ny + 1):
                ax = fig.add_subplot(ny, nx, i)
                img = aruco.drawMarker(aruco_dictionary, i, 2000)

                plt.imshow(img, cmap=plt.cm.gray, interpolation="nearest")
                ax.axis("off")

            plt.savefig("markers.pdf")
            plt.show()
            self.ArucoImage = img

        return self.ArucoImage

    def plot_ir_aruco_location(self, kinect : KinectV2):
        plt.figure(figsize=(20, 20))
        plt.imshow(kinect.get_ir_frame(), cmap="gray")
        plt.plot(self.dict_markers_current["Corners_IR_x"], self.dict_markers_current["Corners_IR_y"], "or")
        plt.show()
        self.ArucoImage = img
        return self.ArucoImage


    def plot_rgb_aruco_location(self, kinect: KinectV2):
        plt.figure(figsize=(20, 20))
        plt.imshow(kinect.get_color())
        plt.plot(self.dict_markers_current["Corners_RGB_x"], self.dict_markers_current["Corners_RGB_y"], "or")
        plt.show()