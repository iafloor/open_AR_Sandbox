import panel as pn
from sandbox import set_logger
from .vlakvergelijking import vlakvergelijking
logger = set_logger(__name__)

class widgets():
    """ Class handles the widgets"""
    def __init__(self):
        pn.extension()
        self.NExercise = -1
        self.start = True
        self.color = False
        self.axes = False
        self.contour = False
        self.params = [False]
        self.plane_equation = False
        self.vector_equation = False
        self.drawPoint = False
        self.random_vector = False
        self.vec_eq = False
        self.drawPoint = False
        self.x = 0
        self.y = 0
        logger.info("widgets created")

    def update(self, sb_params: dict, w_params: dict):
        w_params['Nexercise'] = self.NExercise
        w_params['start'] = self.start
        w_params['params'] = self.params
        w_params['random_vector'] = self.random_vector
        self.random_vector = False
        w_params['vector_equation'] = self.vector_equation
        w_params['x'] = self.x
        w_params['y'] = self.y

        return [sb_params, w_params]
        
    def widgets_test(self):
        self.NExercise = -1
        self._create_widgets()
        panel = pn.Column("Widgets for test",
                        self._widget_x,
                        self._widget_y)
        return panel

    def widgets_tutorial(self):
        self.NExercise = 0
        self.start = True
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 1",
                          self._widget_start)
        return panel

    def widgets_exercise1(self):
        self.NExercise = 1
        self.start = True
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 1",
                          self._widget_start)
        return panel

    def widgets_exercise2(self):
        self.NExercise = 2
        self.start = True
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 2",
                          self._widget_start)
        return panel

    def widgets_exercise3(self):
        self.NExercise = 3
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 3",
                          self._widget_start)
        return panel

    def widgets_exercise4(self):
        self.NExercise = 4
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 4",
                          self._widget_start,
                          self._widget_random_vector,
                          self._widget_vec_eq)
        return panel

    def widgets_exercise5(self):
        self.NExercise = 5
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 5",
                          self._widget_start)
        return panel

    def widgets_exercise6(self):
        self.NExercise = 6
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 6",
                          self._widget_start)
        return panel

    def widgets_exercise7(self):
        self.NExercise = 7
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 7",
                          self._widget_start)
        return panel

    def widgets_exercise8(self):
        self.NExercise = 1
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 8",
                          self._widget_start)
        return panel

    def widgets_exercise9(self):
        self.NExercise = 8
        self._create_widgets()
        panel = pn.Column("### Widgets for exercise 9",
                          self._widget_start)
        return panel

    def _create_widgets(self):
        """
           Create and show the widgets associated to this module
           Returns:
               widget
           """
        self._widget_start = pn.widgets.Checkbox(name="Show exercise description", value=self.start)
        self._widget_start.param.watch(self._callback_start, 'value', onlychanged=True)

        self._widget_random_vector = pn.widgets.Button(name="Show a vector", button_type="primary")
        self._widget_random_vector.param.watch(self._callback_random_vector, 'value', onlychanged=True)

        self._widget_color = pn.widgets.Checkbox(name='Show colors', value=self.color)
        self._widget_color.param.watch(self._callback_color, 'value', onlychanged=False)

        self._widget_contour = pn.widgets.Checkbox(name='Show contours', value=self.contour)
        self._widget_contour.param.watch(self._callback_contour, 'value', onlychanged=False)

        self._widget_axes = pn.widgets.Checkbox(name='Show axes', value=self.axes)
        self._widget_axes.param.watch(self._callback_axes, 'value', onlychanged=False)

        self._widget_plane_eq = pn.widgets.Checkbox(name='Find plane equation', value=self.plane_equation)
        self._widget_plane_eq.param.watch(self._callback_plane_eq, 'value', onlychanged=False)

        self._widget_vec_eq = pn.widgets.Checkbox(name='Find vector equation', value=self.vector_equation)
        self._widget_vec_eq.param.watch(self._callback_vec_eq, 'value', onlychanged=False)

        self._widget_rand_eq = pn.widgets.Button(name='get random equation', button_type='primary')
        self._widget_rand_eq.param.watch(self._callback_equation, 'value', onlychanged=False)

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

        self._widget_point = pn.widgets.Checkbox(name='draw point', value=self.drawPoint)
        self._widget_point.param.watch(self._callback_point, 'value', onlychanged=False)

        self._widget_show_red_points = pn.widgets.Button(name='Show red points', button_type='primary')
        self._widget_show_red_points.param.watch(self._callback_show_red_points, 'value', onlychanged=False)

    def _callback_color(self, event): self.color = event.new

    def _callback_contour(self, event): self.contour = event.new

    def _callback_axes(self, event): self.axes = event.new

    def _callback_equation(self, event): self.get_random_equation = event.new

    def _callback_plane_eq(self, event): self.plane_equation = event.new

    def _callback_vec_eq(self, event): self.vector_equation = event.new

    def _callback_x(self, event): self.x = float(event.new)

    def _callback_y(self, event): self.y = float(event.new)

    def _callback_point(self, event): self.drawPoint = event.new

    def _callback_show_red_points(self, event): self.ShowRedPoints = event.new

    def _callback_start(self, event): self.start = event.new

    def _callback_random_vector(self, event): self.random_vector = event.new
