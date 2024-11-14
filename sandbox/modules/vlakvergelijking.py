import matplotlib.pyplot as plt
import matplotlib
import math
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
from sandbox.modules import exercises
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
        self.axesShown = False
        self.findRed = False
        self.plane_equation = False
        self.vector_equation = False
        self.findHigh = False
        self.findEquation = False
        self.ShowRedPoints = False
        self.height = 100
        self.drawPoint = False
        self.red_points = []
        self.x = 100
        self.y = 100
        self.lines = [0]*46
        self.get_random_equation = False
        self.exercises = exercises()

        ## variables for exercises in general
        self.NExercise = 0
        self.start = True

        ## variables for exercise 1
        self.random_vector = False
        ## variables for exercise 2
        logger.info("VlakModules loaded successfully")

    def update(self, sb_params: dict, w_params: dict):

        # if color or contour is false, we want to not show them
        sb_params['color'] = self.color
        sb_params['contourlines'] = self.contour
        frame = sb_params.get('frame')
        extent = sb_params.get('extent')
        ax = sb_params.get('ax')
        cmap = sb_params.get("cmap")
        colors = sb_params['colors']

        self.NExercise = w_params['Nexercise']
        self.start = w_params['start']
        self.color = w_params['color']

        # after sending a request to make a random vector, end the request
        self.random_vector = w_params['random_vector']
        self.vector_equation = w_params['vector_equation']

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
        return [sb_params, w_params]

    def plot(self, frame, ax, colors, cmap, extent):
        if self.NExercise == 0:
            self.plot_tutorial(ax)
        elif self.NExercise == 1:
            self.plot_exercise_1(ax)
        elif self.NExercise == 2:
            self.plot_exercise_2(ax)
        elif self.NExercise == 3:
            self.plot_exercise_3(ax)
        elif self.NExercise == 4:
            self.plot_exercise_4(ax, colors, frame)
        elif self.NExercise == 5:
            self.plot_exercise_5(ax)
        elif self.NExercise == 6:
            self.plot_exercise_6(ax)
        elif self.NExercise == 7:
            self.plot_exercise_7(ax)
        elif self.NExercise == 8:
            self.plot_exercise_8(ax)
        elif self.NExercise == 9:
            self.plot_exercise_9(ax)
        border_x = frame.shape[1]
        border_y = frame.shape[0]
        # add gridlines
        # vertical
        self.plot_axes(ax, border_x, border_y)
        try:
            p = self.p.pop(0)
            p.remove()
        except:
            pass
        if self.drawPoint:
            self.p = ax.plot(self.y, border_y - self.x, marker='o', fontsize=14, color='red', linewidth=1)

        ## show red points
        if self.ShowRedPoints:
            self.ShowRedPoints = False
            self.show_red_points(border_y, ax)

        return frame, ax, cmap, extent

    def plot_axes(self, ax, border_x, border_y):
        if self.axes and not self.axesShown:
            self.axesShown = True
            # horizontal
            for i in range(-6,7):
                self.lines[i+6] = ax.plot([border_x * (i + 6) / 12, border_x * (i + 6) / 12], [0, border_y], marker='o', color='black',
                    linewidth=0.5)
                self.lines[i+19] = ax.annotate(-1 * i, (border_x * (i + 6) / 12 + 2, border_y / 2 - 5), color="black",
                                          rotation=180)
            for i in range(-4, 5):
                self.lines[i+ 30] = ax.plot([0, border_x], [border_y * (i + 4) / 8, border_y * (i + 4) / 8], marker='o', color='black',
                        linewidth=0.5)
                self.lines[i+ 39] =ax.annotate(-1 * i, (border_x / 2 + 2, border_y * (i + 4) / 8 - 5), color="black",
                                              rotation=180)
            self.lines[44] = ax.plot([0, border_y], [border_x / 2, border_x / 2], marker='o', color='black', linewidth=1)
            self.lines[45] = ax.plot([0, border_x], [border_y / 2, border_y / 2], marker='o', color='black', linewidth=1)
        elif not self.axes and self.axesShown:
            try:
                self.axesShown = False
                for i in self.lines:
                    try:
                        l = i.pop(0)
                        l.remove()
                    except:
                        i.remove()
            except:
                pass

    #--------------------------------------------------------------------------
    #                               Exercises
    #--------------------------------------------------------------------------

    def plot_tutorial(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(self.exercises.tutorial(), (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_1(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(self.exercises.exercise_1(),(40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_2(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(
                self.exercises.exercise_2(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_3(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(
                self.exercises.exercise_3(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_4(self, ax, colors, frame):
        ''' This exercise evolves around '''
        self.color = False
        self.contour = False
        self.axes = True
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.axes = False
            self.description = ax.annotate(self.exercises.exercise_4(), (40, 80), fontsize=28, color="black", rotation=180)

        ## print a random vector
        border_x = colors.shape[0]
        border_y = colors.shape[1]
        if self.random_vector:
            self.random_vector = False
            a = [random.randint(-4,4), random.randint(-4,4)]
            b = [random.randint(-4,4) , random.randint(-4,4)]
            vec1 = [border_y - self.detranslate_x(a[0], border_y), border_x - self.detranslate_y(a[1], border_x)]
            vec2 = [border_y - self.detranslate_x(b[0], border_y), border_x - self.detranslate_y(b[1], border_x)]
            try:
                v = self.vec.pop(0)
                v.remove()
            except:
                pass
            self.vec = ax.plot([vec1[0], vec2[0]],
                           [vec1[1], vec2[1]], marker='o',
                           color='red', linewidth=1)

        ## print dynamic vector
        ## find the vector indicated by two red tokens
        if self.vector_equation:
            try:
                red = self.find_color(colors, 'red')[0]
                blue = self.find_color(colors, 'blue')[0]
                points = [red, blue]
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_equation(translated_points, ax, border_y, dummy_vec)
            except:
                pass

    def plot_exercise_5(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.axes = False
            self.description = ax.annotate(
                self.exercises.exercise_5(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True
            self.axes = True

        ## print a random vector
        border_x = colors.shape[0]
        border_y = colors.shape[1]
        if self.random_vector:
            self.random_vector = False
            dummy_vec = [random.randint(-4, 4), random.randint(-4, 4), random.randint(-4, 4)]
            try:
                self.random_equation.remove()
            except:
                pass
            self.random_equation = ax.annotate("Vector: (" + dummy_vec[0] + ", " + dummy_vec[1] + ", " + dummy_vec[2] + ")", (10, 10), color="#bf0707", fontsize=14, rotation=180)

        if self.vector_equation:
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')
            points = self.add_z([a,b], frame)
            translated_points = self.translate(points, border_x, border_y)
            self.calc_vec_equation(translated_points, ax, border_y, dummy_vec, 0)

    def plot_exercise_6(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(
                self.exercises.exercise_6(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_7(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(
                self.exercises.exercise_7(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_8(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate(
                self.exercises.exercise_8(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_9(self, ax):
        self.color = False
        self.contour = False
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.description = ax.annotate(
                self.exercises.exercise_9(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            if self.vector_equation:
                try:
                    red = self.find_color(colors, 'red')
                    if lengt(red) == 2:
                        translated_points = self.translate(red, border_x, border_y)
                        self.calc_vec_equation(translated_points, ax, border_y, dummy_vec, 0)
                    else:
                        print("not enough or too many points found, number of points:" + length(red))
                except:
                    pass
                try:
                    blue = self.find_color(colors, 'blue')
                    if length(blue) == 2:
                        translated_points = self.translate(blue, border_x, border_y)
                        self.calc_vec_equation(translated_points, ax, border_y, dummy_vec, 1)
                    else:
                        print("not enough or too many points found, number of points:" + length(blue))
                except:
                    pass

    def plot_exercise_10(self, ax):
        self.color = False
        self.contour = False
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.description = ax.annotate(
                self.exercises.exercise_9(),
                (40, 80), fontsize=28, color="black", rotation=180)
        else:
            if self.vector_equation:
                try:
                    red = self.find_color(colors, 'red')
                    if lengt(red) == 2:
                        translated_points = self.translate(red, border_x, border_y)
                        self.calc_vec_equation(translated_points, ax, border_y, dummy_vec, 0)
                    else:
                        print("not enough or too many points found, number of points:" + length(red))
                except:
                    pass
                try:
                    blue = self.find_color(colors, 'blue')
                    if length(blue) == 2:
                        translated_points = self.translate(blue, border_x, border_y)
                        self.calc_vec_equation(translated_points, ax, border_y, dummy_vec, 1)
                    else:
                        print("not enough or too many points found, number of points:" + length(blue))
                except:
                    pass

    #--------------------------------------------------------------------------
    #                              Helper Functions
    #--------------------------------------------------------------------------
    def translate_x(self, x, total):
        return round(x*12/total - 6)

    def detranslate_x(self, x, total):
        return round((x+6)*total/12)

    def translate_y(self, y, total):
        return round(y * 8/ total - 4)

    def detranslate_y(self,y,total):
        return round((y + 4) * total / 8)

    def translate_z(self, z, total):
        return round(z * 8 / total - 4)

    def add_z(self, points, frame):
        ''' Add z variable to x,y pair'''
        for i in range(len(points)):
            self.points[i].append(frame[points[i][0], points[i][1]])
        return points

    def translate(self, points, border_x, border_y):
        translated_points = []
        for i in range(len(points)):
            x = self.translate_x(border_x - points[i][1], border_x)
            y = self.translate_y(points[i][0], border_y)
            p = np.array([x, y, self.translate_z(points[i][2] - 100, 100)])
            translated_points.append(p)
        return translated_points
    #--------------------------------------------------------------------------
    #                           Randomizing
    #--------------------------------------------------------------------------

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
                result = result + str(abs(round(equation[i] * 10,1))) + parameters[i]
        result = result + "= 0"
        return result
    
    def find_color(self, colors, color_to_find):
        ''' Currently, we first look for all red colored points (needs some tweeking when using an actual sandbox
            then, we filter through the list and remove all points that are close to each oter (and just leave one
            in the list. This may also need tweeking. Now, new points have to be at least 10 pixels away, this
            could be changed'''

        if color_to_find == 'red':
            rgb = [0,1,2]
        elif color_to_find == 'green':
            rgb = [1,0,2]
        else: # blue
            rgb = [2,0,1]

        points = [] # list of all red points
        key_points = []
        if color_to_find == "red":
            for i in range(colors.shape[0]): # loop through all pixels
                for j in range(colors.shape[1]):
                    if colors[i][j][rgb[0]] > colors[i][j][rgb[1]] * 2 and colors[i][j][rgb[0]] > colors[i][j][rgb[2]] * 2: # if red/green/blue enough, add to list
                        points.append([i,j])
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
                    if abs(i[0] - j[0]) < 10 and abs(i[1] - j[1]) < 10: # if closer to each other than 10 pixels, remove one
                        points[id] = []
            res = [ele for ele in points if ele != []]
        return res

    def find_red(self, colors):
        ''' Currently, we first look for all red colored points (needs some tweeking when using an actual sandbox
            then, we filter through the list and remove all points that are close to each oter (and just leave one
            in the list. This may also need tweeking. Now, new points have to be at least 10 pixels away, this
            could be changed'''
        points = []  # list of all red points
        key_points = []

        for i in range(colors.shape[0]):  # loop through all pixels
            for j in range(colors.shape[1]):
                if colors[i][j][0] > colors[i][j][1] * 2 and colors[i][j][0] > colors[i][j][
                    2] * 2:  # if red enough, add to list
                    points.append([i, j])
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
                    if abs(i[0] - j[0]) < 10 and abs(
                            i[1] - j[1]) < 10:  # if closer to each other than 10 pixels, remove one
                        points[id] = []
            res = [ele for ele in points if ele != []]
        return res

    def show_red_points(self, border_y, ax):
        ''' Show red points on the height map. To make sure we can use this function to show 1, 2 and 3 points. We just try and otherwise through an exception'''
        try:
            self.labelA.remove()
            pa = self.pointA.pop(0)
            pa.remove()
        except:
            pass
        try:
            self.labelB.remove()
            pb = self.pointB.pop(0)
            pb.remove()
        except:
            pass
        try:    
            self.labelC.remove()
            pc = self.pointC.pop(0)
            pc.remove()
        except:
            pass
        try:
            self.labelA = ax.annotate("A", (self.red_points[0][1] + 1, border_y - self.red_points[0][0] + 1), color="#bf0707", fontsize=14, rotation=180)
            self.pointA = ax.plot(self.red_points[0][1], border_y - self.red_points[0][0], marker='o', color='red', linewidth=1)
        except:
            pass
        try: 
            self.labelB = ax.annotate("B", (self.red_points[1][1] + 1, border_y - self.red_points[1][0] + 1), color="#bf0707", fontsize=14, rotation=180)
            self.pointB = ax.plot(self.red_points[1][1], border_y - self.red_points[1][0], marker='o', color='red', linewidth=1)
        except:
            pass
        try: 
            self.labelC = ax.annotate("C", (self.red_points[2][1] + 1, border_y - self.red_points[2][0] + 1), color="#bf0707", fontsize=14, rotation=180)
            self.pointC = ax.plot(self.red_points[2][1], border_y - self.red_points[2][0], marker='o', color='red', linewidth=1)   
        except:
            pass

    def calc_plane_equation(self, translated_points, ax):
        ## find equation
        n = np.cross(np.subtract(translated_points[0], translated_points[2]),
                     np.subtract(translated_points[1], translated_points[2]))
        norm = np.linalg.norm(n)
        normal = n / norm
        equation = Plane(point=translated_points[0].tolist(), normal=normal.tolist()).cartesian()
        result = self.parameters_to_string(equation)
        try:
            self.equation.remove()
        except:
            pass
        self.equation = ax.annotate("Equation:" + result, (100, 3), color="#bf0707", fontsize=14, rotation=180)
        
        ## other format
        equation = Plane(point=translated_points[0].tolist(), normal=normal.tolist()).cartesian()
        z_1_equation = [round(round(equation[0]*10)/round(equation[2]*10),2), round(round(equation[1]*10)/round(equation[2]*10)), 1, round(round(equation[3]*10)/round(equation[2]*10))]
        result = "z = " + str(z_1_equation[0]) + "x +" + str(z_1_equation[1]) + "y +" + str(z_1_equation[3])  
        try:
            self.alt_equation.remove()
        except:
            pass
        self.alt_equation = ax.annotate("Equation:" + result, (100, 10), color="#bf0707", fontsize=14, rotation=180)
        
    def calc_vec_equation(self, translated_points, ax, border_y, dummy_vec, ind):
        ## find vector representation
        vec = translated_points[1] - translated_points[0]
        result = "(" + str(vec[0]) + ", " + str(vec[1]) + ", " + str(vec[2]) + ")"
        try: 
            e = self.vec_equation.pop(0)
            e.remove()
        except:
            pass
        if vec == dummy_vec:
            color="#09db3d"
        else:
            color="#bf0707"
        self.vec_equation = ax.annotate("Vector:" + result, (100 + ind*10,3), color=color, fontsize=14, rotation=180)
        
        ## print the vector
        try:
            v = self.vec.pop(0)
            v.remove()
        except:
            pass
            
        self.vec = ax.plot([self.red_points[1][1], self.red_points[1][1]], [border_y - self.red_points[1][0], border_y - self.red_points[1][0]+ 10] , marker='o', color='red', linewidth=1)

    def show_widgets(self):
        pass