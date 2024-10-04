import matplotlib.pyplot as plt
import matplotlib
import random
from matplotlib.colors import LightSource
import numpy
import panel as pn
import numpy as np
import pandas as pd
import csv
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
        self.findRed = False
        self.findHigh = False
        self.findEquation = False
        self.height = 100
        self.x = 100
        self.y = 100
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
        print("borders", border_x, border_y)
        # add gridlines
        # vertical
        if self.axes:
            for i in range(-6,7):
                ax.plot([border_x*(i+6)/12, border_x*(i+6)/12], [0, border_y], marker='o', color='black', linewidth=0.5)
                self.axeslabels = ax.annotate(-1*i, (border_x*(i+6)/12 + 2,border_y/2-5), color="black", rotation=180)
            ax.plot([border_x/2, border_x/2], [0, border_y], marker='o', color='black', linewidth=1)

            # horizontal
            for i in range(-4,5):
                ax.plot([0, border_x],[border_y * (i + 4)/8, border_y * (i + 4)/8], marker='o', color='black', linewidth=0.5)
                self.axeslabels = ax.annotate(-1*i, (border_x / 2 + 2, border_y * (i + 4) / 8 - 5), color="black", rotation=180)
            ax.plot([0, border_x], [border_y /2, border_y /2], marker='o', color='black', linewidth=1)
       
        try:
            p = self.p.pop(0)
            p.remove()
        except:
            pass
        self.p = ax.plot(self.x, self.y, marker='o', color='red', linewidth=1)
        
        
        if self.findRed:
            
            ## rgb to hex 
            '''
            colorshex = [["" for i in range(colors.shape[1])] for j in range(colors.shape[0])]
            print("shape hex", len(colorshex))
            for i in range(len(colorshex)):
                for j in range(len(colorshex[0])):
                    res = str("#{0:02x}{1:02x}{2:02x}".format(int(colors[i][j][0]),int(colors[i][j][1]),int(colors[i][j][2])))
                    colorshex[i][j] = res[1:]
            with open("bar.csv", 'w') as resultFile:
                wr = csv.writer(resultFile, dialect='excel', delimiter =';')
                wr.writerows(colorshex)  
            '''
            
            red_points = self.find_red(colors)
            if len(red_points) == 1:
                print("there should be a red point now")
                try:
                    p = self.Redpoint.pop(0)
                    p.remove()
                except:
                    pass
                self.Redpoint = ax.plot(red_points[0][1], border_y - red_points[0][0], marker='o', color='red', linewidth=1)
                
            if self.findEquation:
                self.findEquation = False
                ## add z coordinate
                frame1 = frame.shape[1]
                frame0 = frame.shape[0]
                color1 = colors.shape[1]
                color0 = colors.shape[0]
                for i in range(len(red_points)):

                    ## translate from color to frame
                    id1 = round(frame1 * red_points[i][1]/color1) - 1
                    id0 = round(frame0 * red_points[i][0]/color0) - 1
                    height = frame[id0,id1]
                    red_points[i].append(height)
                    red_points[i][0] = frame1 - id1
                    red_points[i][1] = id0
                    print("the point", red_points)

                ## print the points and labels
                labels = ["A", "B", "C"]
                try:
                    self.labelA.remove()
                    pa = self.pointA.pop(0)
                    pa.remove()
                    self.labelB.remove()
                    pb = self.pointB.pop(0)
                    pb.remove()
                    self.labelC.remove()
                    pc = self.pointC.pop(0)
                    pc.remove()
                except:
                    pass
                self.labelA = ax.annotate("A", (red_points[0][0] + 1, red_points[0][1] + 1), color="#bf0707", fontsize=14, rotation=180)
                self.labelB = ax.annotate("B", (red_points[1][0] + 1, red_points[1][1] + 1), color="#bf0707", fontsize=14, rotation=180)
                self.labelC = ax.annotate("C", (red_points[2][0] + 1, red_points[2][1] + 1), color="#bf0707", fontsize=14, rotation=180)
                self.pointA = ax.plot(red_points[0][0], red_points[0][1], marker='o', color='red', linewidth=1)
                self.pointB = ax.plot(red_points[1][0], red_points[1][1], marker='o', color='red', linewidth=1)
                self.pointC = ax.plot(red_points[2][0], red_points[2][1], marker='o', color='red', linewidth=1)

                ## find coëfficients of plane through min max and 50,50
                translated_points = []
                for i in range(3):
                    p = np.array([self.translate_x(red_points[i][0], border_x), self.translate_y(red_points[i][1], border_y), self.translate_z(red_points[i][2], 100)])
                    translated_points.append(p)

                ## finding the equation
                self.plane_equation(translated_points, ax)

        if self.findHigh:
            print("frame size", frame.shape)
            high_points = self.find_high(frame, self.height)
            print("all high points", high_points)
        ## random plane
        if self.get_random_equation:
            self.create_random_plane_equation(ax)


        return frame, ax, cmap, extent

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
        self.random_equation = ax.annotate("Random equation:" + result, (10, 10), color="#bf0707", fontsize=14, rotation=180)

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

    def reshape_colors(self, colors, frame):
        ''' function reshapes color frame to be the same size as colors'''
        width_f = frame.shape[0]
        height_f = frame.shape[1]
        
        width_c = colors.shape[0]
        height_c = colors.shape[1]
        
        ratio_x = width_c/width_f
        ratio_y = height_c/ height_f
        
        print("ratios", ratio_x, ratio_y)
        new_colors = np.zeros([frame.shape[0], frame.shape[1], 3])
        print(new_colors.shape)
        
        for i in range(new_colors.shape[0]):
            for j in range(new_colors.shape[1]):
                new_colors[i][j] = colors[round(i*ratio_x)][round(j*ratio_y)]
        return new_colors

        
        
    ## FUNCTION TO FIND ALL RED POINTS
    def find_red(self, colors):
        ''' Currently, we first look for all red colored points (needs some tweeking when using an actual sandbox
            then, we filter through the list and remove all points that are close to each oter (and just leave one
            in the list. This may also need tweeking. Now, new points have to be at least 10 pixels away, this
            could be changed'''
        points = [] # list of all red points
        key_points = []
        print("searching for red points")
        
        
        for i in range(colors.shape[0]): # loop through all pixels
            for j in range(colors.shape[1]):
                if colors[i][j][0] > 200 and colors[i][j][1] < 150 and colors[i][j][2] < 150: # if red enough, add to list
                    points.append([i,j])

        print("len points", len(points))
        print("all red points", points)
        ## find key points
        #  for i in points:
        res = []
        for i in points:
            if i == []:
                continue
            for id, j in enumerate(points):
                if i == j or i == [] or j == []:
                   continue
                else:
                    if abs(i[0] - j[0]) < 40 and abs(i[1] - j[1]) < 40: # if closer to each other than 10 pixels, remove one
                        points[id] = []
            res = [ele for ele in points if ele != []]
        print(res)
        # if we have found 3 points, we can make an equation
        if len(res) == 1:
            print("found one red point, no equation made")
            self.findRed = False
        if len(res) == 3:
            print("found three red points")
            self.findEquation = True
            self.findRed = False
        return res

    def find_high(self, frame, height):
        ''' function to find all high points, needed for calibration.'''
        print("height", height)
        print("frame", frame[0][0])
        df = pd.DataFrame(frame)
        df = df.astype(float).round(3)
        df.to_csv("foo.csv", sep=';', header=False)
        high_points = []
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if frame[i][j] > height:
                    high_points.append([i,j])
                    
        print("high_points", high_points)
        self.findHigh = False
    def plane_equation(self, translated_points, ax):
        ## find equation
        n = np.cross(np.subtract(translated_points[0], translated_points[2]),
                     np.subtract(translated_points[1], translated_points[2]))
        norm = np.linalg.norm(n)
        normal = n / norm
        equation = Plane(point=translated_points[0].tolist(), normal=normal.tolist()).cartesian()
        result = self.parameters_to_string(equation)
        try:
            self.equation .remove()
        except:
            pass
        self.equation = ax.annotate("Equation:" + result, (100, 3), color="#bf0707", fontsize=14, rotation=180)
        
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

        self._widget_red = pn.widgets.Button(name='Find red points', button_type='primary')
        self._widget_red.param.watch(self._callback_find_red, 'value', onlychanged=False)

        self._widget_high = pn.widgets.Button(name='Find high points', button_type='primary')
        self._widget_high.param.watch(self._callback_find_high, 'value', onlychanged=False)

        self._widget_rand_eq = pn.widgets.Button(name='get random equation', button_type='primary')
        self._widget_rand_eq.param.watch(self._callback_equation, 'value', onlychanged=False)
        
        self._widget_high_param = pn.widgets.IntSlider(name='Height',
                                                  bar_color="#0000ff",
                                                  value=100,
                                                  start=1,
                                                  end=400)
        self._widget_high_param.param.watch(self._callback_set_height, 'value', onlychanged=False) 

        self._widget_x = pn.widgets.IntSlider(name='x',
                                                  bar_color="#0000ff",
                                                  value=100,
                                                  start=1,
                                                  end=400)
        self._widget_x.param.watch(self._callback_x, 'value', onlychanged=False)

        self._widget_y = pn.widgets.IntSlider(name='y',
                                                  bar_color="#0000ff",
                                                  value=100,
                                                  start=1,
                                                  end=400)
        self._widget_y.param.watch(self._callback_y, 'value', onlychanged=False)        
        
        self._widget_point = pn.widgets.Button(name='draw point', button_type='primary')
        self._widget_point.param.watch(self._callback_point, 'value', onlychanged=False)

    def show_widgets(self):
        self._create_widgets()
        panel = pn.Column("### Widgets for vlak vergelijking",
                          self._widget_color,
                          self._widget_contour,
                          self._widget_axes,
                          self._widget_red,
                          self._widget_high_param,
                          self._widget_high,
                          self._widget_rand_eq,
                          self._widget_x,
                          self._widget_y,
                          self._widget_point
                          )
        return panel

    def _callback_color(self, event): self.color = event.new

    def _callback_contour(self, event): self.contour = event.new

    def _callback_axes(self, event): self.axes = event.new

    def _callback_equation(self, event): self.get_random_equation = event.new

    def _callback_find_red(self, event): self.findRed = event.new

    def _callback_find_high(self, event): self.findHigh = event.new
    
    def _callback_set_height(self, event) : self.height = float(event.new)
    
    def _callback_x(self, event) : self.x = float(event.new)
    
    def _callback_y(self, event) : self.y = float(event.new)
    
    def _callback_point(self,event) : self.drawPoint = event.new
