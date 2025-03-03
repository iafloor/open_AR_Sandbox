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
        self.original_frame = None
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
        self.point = False
        self.x = 100
        self.y = 100
        self.lines = [0]*46
        self.get_random_equation = False
        self.exercises = exercises()
        self.initialize_depth = True
        self.depth_array = None

        ## variables for exercises in general
        self.NExercise = -1
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
        self.x = w_params['x']
        self.y = w_params['y']
        
        # first time running the code we make depth frame
        if self.initialize_depth:
            self.find_depth(ax, frame)

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
        ''' Based on value of self.NExercise, the functionality for an exercise is created.
            The value of self.NExercise is set based on what exercise is selected in jupyter notebook. 
            Self.NExercise is imported from widgets.py. self.exercise.opdracht_x is imported from the file exercises.py.
            The exercise description is in Dutch.'''
        if self.NExercise == 0:
            self.plot_tutorial(ax, colors, frame)
        elif self.NExercise == 1:
            self.plot_exercise_1(ax, colors, frame, self.exercises.opdracht_1())
        elif self.NExercise == 2:
            self.plot_exercise_2(ax, colors, frame, self.exercises.opdracht_2())
        elif self.NExercise == 3:
            self.plot_exercise_3a(ax, colors, frame, self.exercises.opdracht_3a())
        elif self.NExercise == 4:
            self.plot_exercise_3b(ax, colors, frame, self.exercises.opdracht_3b())
        elif self.NExercise == 5:
            self.plot_exercise_4(ax, colors, frame, self.exercises.opdracht_4a())
        elif self.NExercise == 6:
            self.plot_exercise_4(ax, colors, frame, self.exercises.opdracht_4b())
        elif self.NExercise == 7:
            self.plot_exercise_4(ax, colors, frame, self.exercises.opdracht_4c())
        elif self.NExercise == 8:
            self.plot_exercise_5(ax, colors, frame, self.exercises.opdracht_5())
        elif self.NExercise == 9:
            self.plot_exercise_6(ax, colors, frame, self.exercises.opdracht_6())
        else:
            self.plot_test(ax, colors)
        border_x = frame.shape[1]
        border_y = frame.shape[0]
        # add gridlines
        # vertical
        #self.plot_axes(ax, border_x, border_y)

        ## show red points
        if self.ShowRedPoints:
            self.ShowRedPoints = False
            self.show_red_points(border_y, ax)

        return frame, ax, cmap, extent

    def plot_axes(self, ax, border_x, border_y):
        ''' This function plots the lines of the coordinate system. To be able to delete them later, we need to store the
            lines at a specific location: in the list self.lines. Lines are being drawn by using the function ax.plot, text
            is printed by using the function ax.annotate. See documentation of library matplotlib.axes for more information.'''

        # boolean to indicate that the coordinate system is being printed
        self.axesShown = True

        # horizontal lines
        for i in range(-6,7):
            self.lines[i+6] = ax.plot([border_x * (i + 6) / 12, border_x * (i + 6) / 12], [0, border_y], marker='o', color='gray',
                linewidth=0.5)
            self.lines[i+19] = ax.annotate(-1 * i, (border_x * (i + 6) / 12 + 2, border_y / 2 - 5), color="gray",
                                    rotation=180)
        # vertical lines
        for i in range(-4, 5):
            self.lines[i+ 30] = ax.plot([0, border_x], [border_y * (i + 4) / 8, border_y * (i + 4) / 8], marker='o', color='gray',
                    linewidth=0.5)
            self.lines[i+ 39] =ax.annotate(-1 * i, (border_x / 2 + 2, border_y * (i + 4) / 8 - 5), color="gray",
                                          rotation=180)
    # file needs to include function show_widgets, but the widgets are created in the file widgets.py, thus this function
    # is empty
    def show_widgets():
        pass
    #--------------------------------------------------------------------------
    #                               Exercises
    #--------------------------------------------------------------------------

    def plot_test(self, ax, colors):        
        # use code below when difficulties calibrating color sensor
        colorshex = [["" for i in range(colors.shape[1])] for j in range(colors.shape[0])]
        print("shape hex", len(colorshex))
        for i in range(len(colorshex)):
            for j in range(len(colorshex[0])):
                res = str("#{0:02x}{1:02x}{2:02x}".format(int(colors[i][j][0]),int(colors[i][j][1]),int(colors[i][j][2])))
                colorshex[i][j] = res[1:]

        # all colors are written to a csv file, this file can be opend in excel to see what color every cell has and to check
        # if the color sensor works correctly
        with open("bar.csv", 'w') as resultFile:
            wr = csv.writer(resultFile, dialect='excel', delimiter =';')
            wr.writerows(colorshex)

        ax.cla()    
        border = colors.shape[0]

        # functions to check red and blue points are found
        red = self.find_color(colors, 'red')
        print("red", red)
        blue = self.find_color(colors, 'blue')
        print("blue", blue)

        # delete previous point
        try:
            pa = self.pointA.pop(0)
            pa.remove()
        except:
            pass

        # find a red point and print a red dot at this location. This is done to check if the projection and the color
        # sensor are aligned. If the red dot is not printed on the red point, they are not aligned.
        try:
            red = self.find_color(colors, 'red')
            r = red[0]
            self.pointA = ax.plot(r[1], border - r[0], marker='o', color='blue', linewidth=1)
        except:
            pass
       
    def plot_tutorial(self, ax, colors, frame):
        ax.cla()
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.color = False
            self.contour = False
            self.description = ax.annotate(self.exercises.tutorial(), (40, 40), fontsize=28, color="black", rotation=180)
        else:
            self.color = True
            self.contour = True

    def plot_exercise_1(self, ax, colors, frame, question):
        ''' This function includes functionality to recieve the symbol feedback on a vector '''

        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.description = ax.annotate(question, (40, 60), fontsize=28, color="black", rotation=180)
        else:
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])
            
            # find two red points
            # write vector 
            ## print a random vector
            border_y = colors.shape[0]
            border_x = colors.shape[1]


            # initialize lists
            red = []
            blue = []

            # find all red and blue points (there should be one of each)
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')

            # calculate and print the vector if one blue and one red point is found
            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_equation(translated_points, points, ax, border_y, 1)      

            # if more/less red/blue points are found, we can't create a vector
            else:
                pass

    def plot_exercise_2(self, ax, colors, frame, question):
        ''' This funcion includes functionality to show the description of a given vector at the top of the sandbox.
            In addition, such as in exercise 1, the symbol feedback is shown for one vector created by a blue and red point
            in the sandbox.'''
        self.color = False
        self.contour = False

        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.axes = False
            self.description = ax.annotate(question,(40, 60), fontsize=28, color="black", rotation=180)
        else:

            # We construct a random vector for the student to recreate
            border_y = colors.shape[0]
            border_x = colors.shape[1]
            
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])
            
            if self.random_vector:
                self.random_vector = False
                self.dummy_vec = [random.randint(-2, 2), random.randint(-2, 2), random.randint(1, 3)]
            try:   
                self.random_equation = ax.annotate("Vector: (" + str(self.dummy_vec[0]) + ", " + str(self.dummy_vec[1]) + ", " + str(self.dummy_vec[2]) + ")", (100, 5), color="#bf0707", fontsize=14, rotation=180)
            except:
                pass

            # initialize lists
            red = []
            blue = []

            # find all red and blue points (there should be one of each)
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')

            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_equation(translated_points, points, ax, border_y, 1)

            # if more/less red/blue points are found, we can't create a vector
            else:
                pass

    def plot_exercise_3a(self, ax, colors, frame, question):
        ''' This function includes the functionality to show the symbol feedback on the vector representation that
            is created by a blue and a red point. In addition, a random vector representation is printed at the top of
            the sandbox.'''

        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.description = ax.annotate(
                question,
                (40, 60), fontsize=28, color="black", rotation=180)
        else:
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])

            # construct a random vector representation
            if self.random_vector:
                self.random_vector = False
                self.dummy_vec = [random.randint(-2, 2), random.randint(-2, 2), random.randint(1, 3)]
                result = "Maak de vector: t -> (" + str(random.randint(-2,2)) + ", -2, 0) + "
                self.result = result + "(" + str(self.dummy_vec[0]) + ", " + str(self.dummy_vec[1]) + ", " + str(self.dummy_vec[2]) + ")t"
            try:   
                self.random_equation = ax.annotate(self.result, (50, 10), color="#bf0707", fontsize=14, rotation=180)
            except:
                pass
        
            border_y = colors.shape[0]
            border_x = colors.shape[1]
            
            # find red and blue point red points
            red = []
            blue = []
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')
            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_representation(translated_points, points, ax, border_y, 2)

            # if more/less red/blue points are found, we can't create a vector
            else:
                pass

    def plot_exercise_3b(self, ax, colors, frame, question):
        '''This function includes functionality to show two given vector representations and the show symbol feedback of
            one vecotr representation created by a blue and a red point.'''

        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.description = ax.annotate(
                question,
                (40, 60), fontsize=28, color="black", rotation=180)
        else:
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])

            # show two vector representations
            self.result1 = "Vector 1: t1 ->  ( -2, -1, 0) + (1, 1, 1)t1"
            self.result2 = "Vector 2: t2 -> (-2, -1, 0) + (2, 2, 2)t2"  
            self.random_equation1 = ax.annotate(self.result1, (50, 10), color="#bf0707", fontsize=14, rotation=180)
            self.random_equation2 = ax.annotate(self.result2, (50, 20), color="#bf0707", fontsize=14, rotation=180)
        
            border_y = colors.shape[0]
            border_x = colors.shape[1]
            
            # find red and blue point red points
            red = []
            blue = []
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')
            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                print(points)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_representation(translated_points, points, ax, border_y, 3)

            # if more/less red/blue points are found, we can't create a vector
            else:
                pass
                  
    def plot_exercise_4(self, ax, colors, frame, question):
        ''' This function includes functionality to show one given vector representation and to show symbol feedback of
            a vector representation given by a blue and a red point.'''

        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.axes = False
            self.description = ax.annotate(question, (40, 60), fontsize=28, color="black", rotation=180)
        else:
            ## print a random vector
            border_y = colors.shape[0]
            border_x = colors.shape[1]
            
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])
            
            # show given vector
            self.result1 = "Vector 1: t ->  ( -2, -1, 0) + (2, 2, 2)t"
            self.random_equation1 = ax.annotate(self.result1, (50, 10), color="#bf0707", fontsize=14, rotation=180)
            
            # symbol feedback vector representation
            red = []
            blue = []
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')
            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_equation(translated_points, points, ax, border_y, 1)

            # if more/less red/blue points are found, we can't create a vector
            else:
                pass
             
    def plot_exercise_5(self, ax, colors, frame, question):
        ''' This function includes functionality to show symbol feedback of a vector representation created by a
            blue and a red point.'''
        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.description = ax.annotate(question, (40, 60), fontsize=28, color="black", rotation=180)
        else:
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])
               
            border_y = colors.shape[0]
            border_x = colors.shape[1]
            
            # find red and blue point red points
            red = []
            blue = []
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')
            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_representation(translated_points, points, ax, border_y, 2)              
            else:
                pass
    
    def plot_exercise_6(self, ax, colors, frame, question):
        ''' This function includes functionality to calculate the dot product between a given vector and a vector that is
            created by a blue and a red point.'''
        self.color = False
        self.contour = False

        # delete exercise description and clear the coordinate system (this is done for every exercise)
        try:
            self.description.remove()
        except:
            pass
        if self.start:
            ax.cla()
            self.axes = False
            self.description = ax.annotate(question,(40, 60), fontsize=28, color="black", rotation=180)
        else:
            ## print a random vector
            border_y = colors.shape[0]
            border_x = colors.shape[1]
            
            ax.cla()
            self.plot_axes(ax, frame.shape[1], frame.shape[0])
            
            if self.random_vector:
                self.random_vector = False
                #self.dummy_vec = [random.randint(-2, 2), random.randint(-2, 2), random.randint(1, 3)]
            try:   
                self.random_equation = ax.annotate("Vector: (2, 0, 0)", (100, 5), color="#bf0707", fontsize=14, rotation=180)
            except:
                pass
            red = []
            blue = []
            red = self.find_color(colors, 'red')
            blue = self.find_color(colors, 'blue')
            if len(red) == 1 and len(blue) == 1:
                points = self.alternative_add_z([red[0],blue[0]], frame)
                translated_points = self.translate(points, border_x, border_y)
                self.calc_vec_equation(translated_points, points, ax, border_y, 1)             
                vec = translated_points[1] - translated_points[0]

                # dot product
                result = vec[0]*2  + vec[1]*0 + vec[2]*0
                self.dot_product = ax.annotate("Inproduct: " + str(result), (100, 20), color="#bf0707", fontsize=14, rotation=180)

            # if more/less red/blue points are found, we can't create a vector
            else:
                pass
    #--------------------------------------------------------------------------
    #                              Helper Functions
    #--------------------------------------------------------------------------

    # functions to (de)translate a cell from/to frame from/to a relative x and y coordinate
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
            raw = self.original_frame[points[i][0]][points[i][1]] - frame[points[i][0]][points[i][1]]
            points[i].append(round(raw/15))
        return points

    def alternative_add_z(self, points, frame):
        points[1].append(self.find_highest_point(frame, points[1][0],points[1][1]))
        points[0].append(self.find_lowest_point(frame, points[0][0],points[0][1]))
        return points
        
    def translate(self, points, border_x, border_y):
        translated_points = []
        for i in range(len(points)):
            x = self.translate_x(border_x - points[i][1], border_x)
            y = self.translate_y(points[i][0], border_y)
            p = np.array([x, y, points[i][2]])
            translated_points.append(p)
        return translated_points

    # used to approximate the z coordinate
    def find_highest_point(self, frame, x, y):
        highest_z = 0
        for i in range(frame.shape[1] - 40):
            for j in range(40):
                y_t = min(frame.shape[1] - 20, y + j - 20)
                x_t = min(frame.shape[0] - 20 ,i + 20)
                raw = self.original_frame[x_t, y_t] - frame[x_t, y_t]
                if abs(round(raw/15)) > highest_z:
                    highest_z = abs(round(raw/15))
        return highest_z

    # used to approximate the z coordinate
    def find_lowest_point(self,frame,x,y):
        lowest_z = 0
        for i in range(frame.shape[1] - 40):
            for j in range(40):
                y_t = min(frame.shape[1] - 20, y + j - 20)
                x_t = min(frame.shape[0] - 20 ,i + 20)
                raw = self.original_frame[x_t, y_t] - frame[x_t, y_t]
                if abs(round(raw/15)) < lowest_z:
                    lowest_z = abs(round(raw/15))
        return lowest_z
    
    def find_color(self, colors, color_to_find):
        ''' Currently, we first look for all red colored points (needs some tweeking when using an actual sandbox
            then, we filter through the list and remove all points that are close to each oter (and just leave one
            in the list. This may also need tweeking. Now, new points have to be at least 10 pixels away, this
            could be changed'''

        points = [] # list of all red points
        key_points = []
        if color_to_find == "red":
            for i in range(colors.shape[0]): # loop through all pixels
                for j in range(colors.shape[1]):
                     if colors[i][j][0] > 200 and colors[i][j][1] < 150: # if red/green/blue enough, add to list
                        points.append([i,j])
                        
        if color_to_find == "blue":
            for i in range(colors.shape[0]): # loop through all pixels
                for j in range(colors.shape[1]):
                    if colors[i][j][2] > 200 and colors[i][j][0] < 100:
                        points.append([i,j])
                        
        if color_to_find == "green":
            for i in range(colors.shape[0]): # loop through all pixels
                for j in range(colors.shape[1]):
                    if colors[i][j][2] > 240 and colors[i][j][1] > 220 and colors[i][j][0] < 100:
                        points.append([i,j])
                        
        if color_to_find == "dark green":
            for i in range(colors.shape[0]): # loop through all pixels
                for j in range(colors.shape[1]):
                    if colors[i][j][2] < 200 and colors[i][j][2] > 180 and colors[i][j][1] < 170 and colors[i][j][1] > 150 and colors[i][j][0] < 50:
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
                    if abs(i[0] - j[0]) < 15 and abs(i[1] - j[1]) < 15: # if closer to each other than 10 pixels, remove one
                        points[id] = []
            res = [ele for ele in points if ele != []]
        return res
    #--------------------------------------------------------------------------
    #                           Test Functions
    #--------------------------------------------------------------------------
    def general_depth(self, ax, frame):   
        coordinates = []
        high = 0
        coord = [0,0]
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                frame[i][j] = round((self.original_frame[i][j] - frame[i][j])/15)
                if abs(frame[i][j]) > high and i > 20 and j > 20 and i < len(frame) - 20 and j < len(frame[0]) - 20:
                    coord = [i,j]
        print("high point", i,j)
        
        try:
            pa = self.redPoint.pop(0)
            pa.remove()
        except:
            pass
          
        # print new point
        self.redPoint = ax.plot(coord[1],  frame.shape[0] - coord[0], marker='o', color='red', linewidth=1)
        
    def find_depth(self, ax, frame):
        ''' function makes array of all depths of coordinates in coordinate system: 
        find all depths of 'x' in image below.
        
            l    l    l    l    l    l    l    l
            l    l    l    l    l    l    l    l
        ----X----X----X----X----X----X----X----X----
            l    l    l    l    l    l    l    l
            l    l    l    l    l    l    l    l
        ----X----X----X----X----X----X----X----X----
            l    l    l    l    l    l    l    l
            l    l    l    l    l    l    l    l
        ----X----X----X----X----X----X----X----X----
            l    l    l    l    l    l    l    l
            l    l    l    l    l    l    l    l
        ----X----X----X----X----X----X----X----X----
            l    l    l    l    l    l    l    l
            l    l    l    l    l    l    l    l
        ----X----X----X----X----X----X----X----X----
            l    l    l    l    l    l    l    l
            l    l    l    l    l    l    l    l
            '''
        self.initialize_depth = False
        depth_array = []
        self.original_frame = frame
        border_x = frame.shape[1]
        border_y = frame.shape[0]
        for i in range(-5,6):
            depth_array.append([])
            for j in range(-3,4):
                
                
                # find depth for point (i,j)
                x = self.detranslate_x(i, border_x)
                y = self.detranslate_y(j, border_y)
                depth_array[i+5].append(round(frame[y,x]))
                
        # sometimes something goes wrong and all values are set to 324. If this is the case we try again.
        if depth_array[0][0] == 324:
            self.initialize_depth = True
        self.depth_array = depth_array
    
    #--------------------------------------------------------------------------
    #                           Randomizing
    #--------------------------------------------------------------------------
     
    def parameters_to_string(self, equation, roundup):
        ''' Combine the parameters to one string to be printed'''
        parameters = ["x ", "y ", "z ", " "]
        id = 0
        #if equation[id] == 0:
        #    id = 1
        result = str(round(equation[id] * 10)) + parameters[id]

        for i in range(id+1, 4):
            if True: #equation[i] != 0:
                if equation[i] < 0:
                    result = result + "- "
                else:
                    result = result + "+ "
                if roundup:
                    result = result + str(abs(round(equation[i] * 10))) + parameters[i]
                else:
                    result = result + str(abs(round(equation[i] * 10,1))) + parameters[i]
        result = result + "= 0"
        return result
    
    #--------------------------------------------------------------------------
    #                           vec / plane calculations
    #--------------------------------------------------------------------------
    
    def calc_plane_equation(self, translated_points, ax):
        ## find equation
        n = np.cross(np.subtract(translated_points[0], translated_points[2]),
                     np.subtract(translated_points[1], translated_points[2]))
        norm = np.linalg.norm(n)
        normal = n / norm
        return Plane(point=translated_points[0].tolist(), normal=normal.tolist()).cartesian()        
        
    def plane_format_d(self,ax, equation, roundup, ind):
        result = self.parameters_to_string(equation, roundup)
        try:
            self.equation.remove()
        except:
            pass
        self.equation = ax.annotate("Equation:" + result, (100, 10*ind), color="#bf0707", fontsize=14, rotation=180)
    
    def plane_format_z(self, ax, equation, ind):
        z_1_equation = [round(round(equation[0]*10)/round(equation[2]*10)), round(round(equation[1]*10)/round(equation[2]*10)), 1, round(round(equation[3]*10)/round(equation[2]*10))]
        result = "z = " + str(z_1_equation[0]*-1) + "x +" + str(z_1_equation[1]*-1) + "y +" + str(z_1_equation[3]*-1)  
        try:
            self.equation_z.remove()
        except:
            pass
        self.equation_z = ax.annotate("Equation:" + result, (100, 10*ind), color="#bf0707", fontsize=14, rotation=180)
        
    def calc_vec_equation(self, translated_points, points, ax, border_y, ind):
        ## find vector representation
        vec = translated_points[1] - translated_points[0]
        result = "(" + str(vec[0]) + ", " + str(vec[1]) + ", " + str(vec[2]) + ")"
        try: 
            e = self.vec_equation.pop(0)
            e.remove()
        except:
            pass
        if False:
            color="#09db3d"
        else:
            color="#bf0707"
        self.vec_equation = ax.annotate("Vector:" + result, (25, 5 + ind*10), color=color, fontsize=14, rotation=180)
        
        ## print the vector
        if ind == 0: 
            try:
                v = self.vec1.pop(0)
                v.remove()
            except:
                pass        
            #self.vec1 = ax.plot([points[1][1], points[0][1]], [border_y - points[1][0], border_y - points[0][0]] , marker='o', color='red', linewidth=1)
        if ind == 1:
            try:
                v = self.vec2.pop(0)
                v.remove()
            except:
                pass
            #self.vec2 = ax.plot([points[1][1], points[0][1]], [border_y - points[1][0], border_y - points[0][0]] , marker='o', color='blue', linewidth=1)
    
    def calc_vec_representation(self, translated_points, points, ax, border_y, ind):
        vec = translated_points[1] - translated_points[0]    
        start = translated_points[0]
        result = "Vector: t -> (" + str(start[0]) + ", " + str(start[1]) + ", " + str(start[2]) + ") + "
        result = result + "(" + str(vec[0]) + ", " + str(vec[1]) + ", " + str(vec[2]) + ")t"
        try: 
            e = self.vec_rep.pop(0)
            e.remove()
        except:
            pass
        self.vec_rep = ax.annotate(result, (50, ind*10), color="#bf0707", fontsize=14, rotation=180)
               
    #--------------------------------------------------------------------------
    #                           not in use
    #--------------------------------------------------------------------------
    
    def show_red_points(self, border_y, ax): # not in use
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