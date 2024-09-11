import matplotlib.pyplot as plt
import matplotlib
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
        self.contour = True
        logger.info("VlakModules loaded successfully")

    def update(self, sb_params: dict):

        # if color or contour is false, we want to not show them
        sb_params['color'] = self.color
        sb_params['contourlines'] = self.contour
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        cmap = sb_params.get("cmap")

        frame, ax, cmap, extent = self.plot(frame, ax, cmap, extent)

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


    def plot(self, frame, ax, cmap, extent):

        if not self.contour:
            # remove contours
            print("remove contours")
            [coll.remove() for coll in reversed(ax.collections) if isinstance(coll, matplotlib.collections.PathCollection)]


        border_x = frame.shape[1]
        border_y = frame.shape[0]
        print("borders", border_x, border_y)

        # add gridlines
        # vertical
        for i in range(-6,7):
            ax.plot([border_x*(i+6)/12, border_x*(i+6)/12], [0, border_y], marker='o', color='black', linewidth=0.5)
            self.text = ax.annotate(i, (border_x*(i+6)/12 + 2,border_y/2-5), color="black")
        ax.plot([border_x/2, border_x/2], [0, border_y], marker='o', color='black', linewidth=1)

        # horizontal
        for i in range(-4,5):
            ax.plot([0, border_x],[border_y * (i + 4)/8, border_y * (i + 4)/8], marker='o', color='black', linewidth=0.5)
            self.text = ax.annotate(i, (border_x / 2 + 2, border_y * (i + 4) / 8 - 5), color="black")
        ax.plot([0, border_x], [border_y /2, border_y /2], marker='o', color='black', linewidth=1)

        # add points based on color
        # find indices of max and min

        shape = frame.shape
        maxid = np.argmax(np.array(frame))
        minid = np.argmin(np.array(frame))

        # transform to x and y id
        maxid_x = int(maxid / shape[1])
        maxid_y = int(maxid % shape[1])
        minid_x = int(minid / shape[1])
        minid_y = int(minid % shape[1])

        # adding text to image

        # adding text to image
        # self.line = ax.plot([minid_y, maxid_y], [minid_x, maxid_x], marker='o', color='white', linewidth=1)
        self.text = self.add_text(ax, (minid_y + 1), (minid_x + 1), "A")
        self.text = self.add_text(ax, (maxid_y - 4), (maxid_x - 4), "B")
        self.text = self.add_text(ax, 101, 101, "C")

        # points
        self.point = ax.plot(minid_y, minid_x, marker='o', color='red', linewidth=1)
        self.point = ax.plot(maxid_y, maxid_x, marker='o', color='red', linewidth=1)
        self.point = ax.plot(100, 100, marker='o', color='red', linewidth=1)
        # find coëfficients of plane through min max and 50,50

        A = np.array([minid_x, minid_y, frame[minid_x][minid_y]])
        B = np.array([maxid_x, maxid_y, frame[maxid_x, maxid_y]])
        C = np.array([100, 100, frame[100][100]])
        n = np.cross(np.subtract(A, C), np.subtract(B, C))
        norm = np.linalg.norm(n)
        normal = n / norm
        equation = Plane(point=A.tolist(), normal=normal.tolist()).cartesian()
        result = "Equation: " + str(round(equation[0], 2)) + "x + " + str(round(equation[1], 2)) + "y + " + str(
            round(equation[2], 2)) + "z + " + str(round(equation[3], 2)) + "= 0"
        self.equation = ax.annotate(result, (3, 3), color="red")
        return frame, ax, cmap, extent

    def add_text(self, ax, x, y, text):
        self.text = ax.annotate(text, (x,y), color='red')

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

    def show_widgets(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for vlak vergelijking",
                          self._widget_color,
                          self._widget_contour
                          )
        return panel

    def _callback_color(self, event): self.color = event.new

    def _callback_contour(self, event): self.contour = event.new
