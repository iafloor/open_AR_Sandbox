import matplotlib.pyplot as plt
import matplotlib
import random
from matplotlib.colors import LightSource
import numpy
import panel as pn
import numpy as np
from skspatial.objects import Plane
from .template import ModuleTemplate
from sandbox import set_logger
logger = set_logger(__name__)


class vlakvergelijking(ModuleTemplate):
    """
    Module to display the gradient of the topography and the topography as a vector field.
    """
    def __init__(self, extent: list = None):
        # call parents' class init, use greyscale colormap as standard and extreme color labeling
        pn.extension()
        if extent is not None:
            self.vmin = extent[4]
            self.vmax = extent[5]

        self.extent = extent
        self.frame = None
        self.color = True
        self.contour = False
        self.axes = False
        self.get_random_equation = False
        logger.info("VlakModules loaded successfully")

    def update(self, sb_params: dict):

        # if color or contour is false, we want to not show them
        sb_params['color'] = self.color
        sb_params['contourlines'] = self.contour
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        cmap = sb_params.get("cmap")
        colors = sb_params['colors']

        frame, ax, cmap, extent = self.plot(frame, ax, colors, cmap, extent)

        sb_params['frame'] = frame
        sb_params['ax'] = ax
        sb_params['cmap'] = cmap
        sb_params['extent'] = extent
        if cmap is None:
            sb_params['active_cmap'] = False
            sb_params['active_shading'] = False
        else:
            sb_params['active_cmap'] = True
            sb_params['active_shading'] = False
        return sb_params


    def plot(self, frame, ax, colors, cmap, extent):

        border_x = frame.shape[1]
        border_y = frame.shape[0]

        # add gridlines
        # vertical
        if self.axes:
            for i in range(-6,7):
                ax.plot([border_x*(i+6)/12, border_x*(i+6)/12], [0, border_y], marker='o', color='black', linewidth=0.5)
                self.text = ax.annotate(-1*i, (border_x*(i+6)/12 + 2,border_y/2-5), color="black", rotation=180)
            ax.plot([border_x/2, border_x/2], [0, border_y], marker='o', color='black', linewidth=1)

            # horizontal
            for i in range(-4,5):
                ax.plot([0, border_x],[border_y * (i + 4)/8, border_y * (i + 4)/8], marker='o', color='black', linewidth=0.5)
                self.text = ax.annotate(-1*i, (border_x / 2 + 2, border_y * (i + 4) / 8 - 5), color="black", rotation=180)
            ax.plot([0, border_x], [border_y /2, border_y /2], marker='o', color='black', linewidth=1)

        red_points = self.find_red(colors)
        if len(red_points) == 3:
            ## add z coordinate
            for i in range(3):
                height = frame[red_points[i][1], red_points[i][0]]
                red_points[i].append(height)

            ## print the points and labels
            labels = ["A", "B", "C"]
            for i in range(3):
                self.text = self.add_text(ax, red_points[i][0] + 1, red_points[i][1] + 1, labels[i] )
                self.point = ax.plot(red_points[i][0], red_points[i][1], marker='o', color='red', linewidth=1)

            ## find coëfficients of plane through min max and 50,50
            translated_points = []
            for i in range(3):
                p = np.array([self.translate_x(red_points[i][0], border_x), self.translate_y(red_points[i][1], border_y), self.translate_z(red_points[i][2], 100)])
                translated_points.append(p)

            ## plane by user
            self.plane_equation(translated_points, ax)

        ## random plane
        if self.get_random_equation:
            self.create_random_plane_equation(ax)


        return frame, ax, cmap, extent

    def add_text(self, ax, x, y, text):
        self.text = ax.annotate(text, (x,y), color='red', rotation=180)

    def _create_widgets(self):
        """
           Create and show the widgets associated to this module
           Returns:
               widget
           """
        self._widget_color = pn.widgets.Checkbox(name='Show colors', value=self.color)
        self._widget_color.param.watch(self._callback_color, 'value', onlychanged=False)

        self._widget_contour = pn.widgets.Checkbox(name='Show contours', value=self.contour)
        self._widget_contour.param.watch(self._callback_contour, 'value', onlychanged=False)

        self._widget_axes = pn.widgets.Checkbox(name='Show axes', value=self.axes)
        self._widget_axes.param.watch(self._callback_axes, 'value', onlychanged=False)

        self._widget_rand_eq = pn.widgets.Button(name='get random equation', button_type='primary')
        self._widget_rand_eq.param.watch(self._callback_equation, 'value', onlychanged=False)

    def translate_x(self, x, total):
        return round(x*12/total - 6,1)

    def translate_y(self, y, total):
        return round(y * 8/ total - 4,1)

    def translate_z(self, z, total):
        return round(z * 8 / total - 4,1)

    def random_plane_parameters(self):
        return [random.randint(-9,9)/10, random.randint(-9,9)/10, random.randint(-9,9)/10, random.randint(-9,9)/10]

    def create_random_plane_equation(self, ax):
        equation = self.random_plane_parameters()
        self.get_random_equation = False
        result = self.parameters_to_string(equation)
        ## remove the old equation
        try:
            self.random_equation.remove()
        except:
            pass

        ## print random equation
        self.random_equation = ax.annotate("Random equation:" + result, (3, 3), color="#bf0707", fontsize=14, rotation=180)

    def parameters_to_string(self, equation):
        ''' Combine the parameters to one string to be printed'''
        parameters = ["x ", "y ", "z ", " "]
        id = 0
        if equation[id] == 0:
            id = 1
        result = str(round(equation[id] * 10)) + parameters[id]

        for i in range(id+1, 4):
            if equation[i] != 0:
                if equation[i] < 0:
                    result = result + "- "
                else:
                    result = result + "+ "
                result = result + str(abs(round(equation[i] * 10))) + parameters[i]
        result = result + "= 0"
        return result

    def find_red(self, colors):
        ''' Currently, we first look for all red colored points (needs some tweeking when using an actual sandbox
            then, we filter through the list and remove all points that are close to each oter (and just leave one
            in the list. This may also need tweeking. Now, new points have to be at least 10 pixels away, this
            could be changed'''
        points = [] # list of all red points
        key_points = []
        for i in range(colors.shape[0]): # loop through all pixels
            for j in range(colors.shape[1]):
                if colors[i][j][0] > 200 and colors[i][j][1] < 100 and colors[i][j][2] < 100: # if red enough, add to list
                    points.append([i,j])

        ## find key points
        for i in points:
            if i == []:
                continue
            for id, j in enumerate(points):
                if i == j or i == [] or j == []:
                   continue
                else:
                    if abs(i[0] - j[0]) < 10 and abs(i[1] - j[1]) < 10: # if closer to each other than 10 pixels, remove one
                        points[id] = []
        res = [ele for ele in points if ele != []]
        return res

    def plane_equation(self, translated_points, ax):
        ## find equation
        n = np.cross(np.subtract(translated_points[0], translated_points[2]),
                     np.subtract(translated_points[1], translated_points[2]))
        norm = np.linalg.norm(n)
        normal = n / norm
        equation = Plane(point=translated_points[0].tolist(), normal=normal.tolist()).cartesian()
        result = self.parameters_to_string(equation)
        self.equation = ax.annotate("Equation:" + result, (100, 3), color="#bf0707", fontsize=14, rotation=180)

    def show_widgets(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for vlak vergelijking",
                          self._widget_color,
                          self._widget_contour,
                          self._widget_axes,
                          self._widget_rand_eq
                          )
        return panel

    def _callback_color(self, event): self.color = event.new

    def _callback_contour(self, event): self.contour = event.new

    def _callback_axes(self, event): self.axes = event.new

    def _callback_equation(self, event): self.get_random_equation = event.new
