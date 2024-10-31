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
        self.get_random_equation = False

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
        self.random_vector = False
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
        else:
            self.plot_exercise_4(ax, colors)
        border_x = frame.shape[1]
        border_y = frame.shape[0]
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
        else:
            try:
                ax.plot.remove()
            except:
                pass
        try:
            p = self.p.pop(0)
            p.remove()
        except:
            pass
        if self.drawPoint:
            self.p = ax.plot(self.y, border_y - self.x, marker='o', fontsize=14, color='red', linewidth=1)
        
        ## find the plane indicated by three red tokens
        if self.plane_equation:
            ## first we need to find the red points
            self.red_points = self.find_red(colors)
            print(self.red_points)
            
            if len(self.red_points) == 1:
                self.red_points[0].append(frame[self.red_points[0][0],self.red_points[0][1]])
                x = self.translate_x(border_x - self.red_points[0][1], border_x)
                y = self.translate_y(self.red_points[0][0], border_y)
                p = np.array([x,y, self.translate_z(self.red_points[0][2], 300)])
                
                df = pd.DataFrame(frame)
                df = df.astype(float).round(3)
                df.to_csv("foo.csv", sep=';', header=False)
                
                            
            ## if there are enough, we can find the equation
            if len(self.red_points) == 3:
            
                ## add z coordinate
                for i in range(len(self.red_points)):
                    self.red_points[i].append(frame[self.red_points[i][0],self.red_points[i][1]])

                ## find coordinates of three red points
                translated_points = []
                for i in range(len(self.red_points)):
                    x = self.translate_x(border_x - self.red_points[i][1], border_x)
                    y = self.translate_y(self.red_points[i][0], border_y)
                    p = np.array([x, y, self.translate_z(self.red_points[i][2], 300)])
                    translated_points.append(p)
                ## finding the equation
                self.calc_plane_equation(translated_points, ax)
                
            ## if not, we print that we did not have enough red points and to try again.
            else:
                if len(self.red_points) > 3:
                    print("too many points found, try again")
                else:
                    print("not enough points found, try again")
                
        ## find the vector indicated by two red tokens
        if self.vector_equation:
        
            ## first we need to find the red points
            self.red_points = self.find_red(colors)
            
            ## if there are enough, we can find the equation
            if len(self.red_points) == 2:
                ## add z coordinate
                for i in range(len(self.red_points)):
                    self.red_points[i].append(frame[self.red_points[i][0],self.red_points[i][1]])

                print("points", self.red_points[0], self.red_points[1])
                ## find coordinates of two red points
                translated_points = []
                for i in range(len(self.red_points)):
                    x = self.translate_x(border_x - self.red_points[i][1], border_x)
                    y = self.translate_y(self.red_points[i][0], border_y)
                    p = np.array([x, y, self.translate_z(self.red_points[i][2] - 100, 100)])
                    translated_points.append(p)
                    
                self.calc_vec_equation(translated_points, ax, border_y)
            elif len(self.red_points) == 1:
                ## add z coordinate
                self.red_points[0].append(frame[self.red_points[0][0], self.red_points[0][1]])
                
                ## add second point right below the first
                self.red_points.append([self.red_points[0][0], self.red_points[0][1],self.red_points[0][2]-10])
            ## if not, we print that we did not have enough red points and to try again.
            else:
                if len(self.red_points) > 2:
                    print("too many points found, try again")
                else:
                    print("not enough points found, try again")
        
        ## show red points
        if self.ShowRedPoints:
            self.ShowRedPoints = False
            self.show_red_points(border_y, ax)
        
        ## random plane
        if self.get_random_equation:
            self.create_random_plane_equation(ax)

        return frame, ax, cmap, extent

    def plot_tutorial(self, ax):
        print(self.NExercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.color = False
            self.contour = False
            self.description = ax.annotate("This is the tutorial.",
                                           (80, 80), fontsize=28, color="black", rotation=180)
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
            self.description = ax.annotate("Move the vector in such a way \n that the height lines on \n the vector disappear.",
                                           (80, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_2(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.description = ax.annotate(
                "Move the vector in such a way \n that the height lines on the vector \n are parallel to the y-axis.",
                (80, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_3(self, ax):
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.description = ax.annotate(
                "Move the vector in such a way \n that the height lines on the vector \n are parallel to the x-axis.",
                (80, 80), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_4(self, ax, colors):
        ''' This exercise evolves around '''
        self.color = False
        self.contour = False
        self.axes = True
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            self.description = ax.annotate("Using the two vector parts, \n make a vector equal \n to the vector AB.", (80, 80), fontsize=28, color="black", rotation=180)

        ## print a random vector
        border_x = colors.shape[0]
        border_y = colors.shape[1]
        if self.random_vector:
            self.random_vector = False
            a = [random.randint(-4,4), random.randint(-4,4)]
            b = [random.randint(-4,4) , random.randint(-4,4)]
            vec1 = [border_y - self.detranslate_x(a[0], border_y), border_x - self.detranslate_y(a[1], border_x)]
            vec2 = [border_y - self.detranslate_x(b[0], border_y), border_x - self.detranslate_y(b[1], border_x)]
            print(a,b)
            print(vec1,vec2)
            try:
                v = self.vec.pop(0)
                v.remove()
            except:
                pass
            self.vec = ax.plot([vec1[0], vec2[0]],
                           [vec1[1], vec2[1]], marker='o',
                           color='red', linewidth=1)

        ## print vector by student
        ## find the vector indicated by two red tokens
        if self.vector_equation:
            self.vector_finding(colors, ax, y-x)

    def translate_x(self, x, total):
        return round(x*12/total - 6)

    def detranslate_x(self, x, total):
        print("x?", x)
        return round((x+6)*total/12)

    def translate_y(self, y, total):
        return round(y * 8/ total - 4)

    def detranslate_y(self,y,total):
        return round((y + 4) * total / 8)

    def translate_z(self, z, total):
        return round(z * 8 / total - 4)

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
    
    def find_red(self, colors):
        ''' Currently, we first look for all red colored points (needs some tweeking when using an actual sandbox
            then, we filter through the list and remove all points that are close to each oter (and just leave one
            in the list. This may also need tweeking. Now, new points have to be at least 10 pixels away, this
            could be changed'''
        points = [] # list of all red points
        key_points = []
        
        for i in range(colors.shape[0]): # loop through all pixels
            for j in range(colors.shape[1]):
                if colors[i][j][0] > colors[i][j][1] * 2 and colors[i][j][0] > colors[i][j][2] * 2: # if red enough, add to list
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
        print(res)
        print("number of distinct red points", len(res))
        return res

    def vector_finding(self, colors, ax, dummy_vec):
        ## first we need to find the red points
        self.red_points = self.find_red(colors)

        ## if there are enough, we can find the equation
        if len(self.red_points) == 2:
            ## add z coordinate
            for i in range(len(self.red_points)):
                self.red_points[i].append(frame[self.red_points[i][0], self.red_points[i][1]])

            print("points", self.red_points[0], self.red_points[1])
            ## find coordinates of two red points
            translated_points = []
            for i in range(len(self.red_points)):
                x = self.translate_x(border_x - self.red_points[i][1], border_x)
                y = self.translate_y(self.red_points[i][0], border_y)
                p = np.array([x, y, self.translate_z(self.red_points[i][2] - 100, 100)])
                translated_points.append(p)

            self.calc_vec_equation(translated_points, ax, border_y, dummy_vec)
            return vec
        elif len(self.red_points) == 1:
            ## add z coordinate
            self.red_points[0].append(frame[self.red_points[0][0], self.red_points[0][1]])

            ## add second point right below the first
            self.red_points.append([self.red_points[0][0], self.red_points[0][1], self.red_points[0][2] - 10])
        ## if not, we print that we did not have enough red points and to try again.
        else:
            if len(self.red_points) > 2:
                print("too many points found, try again")
            else:
                print("not enough points found, try again")
            return [0, 0]

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
        
    def calc_vec_equation(self, translated_points, ax, border_y, dummy_vec):
        ## find vector representation
        vec = translated_points[1] - translated_points[0]
        
        print("vector", vec)
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
        self.vec_equation = ax.annotate("Vector:" + result, (100,3), color=color, fontsize=14, rotation=180)
        
        ## print the vector
        try:
            v = self.vec.pop(0)
            v.remove()
        except:
            pass
            
        self.vec = ax.plot([self.red_points[1][1], self.red_points[1][1]], [border_y - self.red_points[1][0], border_y - self.red_points[1][0]+ 10] , marker='o', color='red', linewidth=1)

    def show_widgets(self):
        pass