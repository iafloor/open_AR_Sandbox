import numpy
import matplotlib
import panel as pn
from sandbox import set_logger
logger = set_logger(__name__)
pn.extension()


class ContourLinesModule:
    dpi = 100  # make sure that figures can be displayed pixel-precise

    def __init__(self, contours=True, contours_step=100,
                 contours_width=1.0, contours_color='k', contours_label=False,
                 contours_label_inline=True, contours_label_fontsize=15,
                 contours_label_format='%3.0f', minor_contours=True,
                 contours_step_minor=50, contours_width_minor=0.5,
                 extent=None, check_change=True, rtol=0.2, atol=0,
                 automatic_levels=False, threshold=200):
        """
            Module for the display and manipulation of contourlines

            Args:
                contours (bool): Flag that enables or disables contours plotting.
                    (default is True)
                contours_step (int): Size of step between contour lines in model units.
                    (default is 100)
                contours_width (float): Width of contour lines.
                    (default is 1.0)
                contours_color (e.g. str): Color of contour lines.
                    (default is 'k')
                contours_label (bool): Flag that enables labels on contour lines.
                    (default is False)
                contours_label_inline (bool): Partly replace underlying contour line or not.
                    (default is True)
                contours_label_fontsize (float or str): Size in points or relative size of contour label.
                    (default is 15)
                contours_label_format (string or dict): Format string for the contour label.
                    (default is %3.0f)
                minor_contours (bool): Flag that enables or disables minor contours plotting.
                    (default is True)
                contours_step_minor (int): Size of step between minor contour lines in model units.
                    (default is 50)
                contours_width_minor (float): Width of minor contour lines.
                    (default is 0.5)
                extent (list): extents of the sandbox to indicate the physical dimensions of it
                check_change (bool): Update the contour ines only when the frame changes
                rtol (float): relative tolerance for checking the change
                atol (float): absolute tolerance for checking the change
                automatic_levels (bool): When changing modules, the extent and data may change, therefore the levels
                    will not remain constant. A quick fix is to set the default levels of matplotlib and let it choose
                    automatically.
                threshold (int): Maximum amount of contour lines per plot. If more then set levels to automatic.
            """
        self.lock = None  # For locking the multithreading while using bokeh server
        self._active = False
        self.active_contours = True
        self.major = None
        self.minor = None
        self.label = None
        self.check_change = check_change
        self._rtol = rtol
        self._atol = atol
        self._threshold = threshold
        self.extent = extent
        self.vmin = self.extent[4]
        self.vmax = self.extent[5]
        self.previous_frame = None

        # flags
        self.contours = contours
        self.minor_contours = minor_contours
        self.automatic_contours = automatic_levels

        # contours setup
        self.contours_step = contours_step  # levels will be supplied via property function
        self.contours_width = contours_width
        self.contours_color = contours_color
        self.contours_label = contours_label
        self.contours_label_inline = contours_label_inline
        self.contours_label_fontsize = contours_label_fontsize
        self.contours_label_format = contours_label_format

        self.contours_step_minor = contours_step_minor
        self.contours_width_minor = contours_width_minor
        logger.info("ContourLinesModule loaded successfully")

    def update(self, sb_params: dict):
        active = sb_params.get('active_contours')
        active = bool(active * self.active_contours)
        ax = sb_params.get('ax')
        frame = sb_params.get('frame')
        if len(frame.shape) > 2:  # 3 Then is an image
            active = False

        if active:
            self._active = active

            if self.previous_frame is None:
                self.previous_frame = frame

            same_frame = sb_params['same_frame']
            if same_frame:
                frame = self.previous_frame
            else:
                self.previous_frame = frame

            extent = sb_params.get('extent')
            self.vmin = extent[-2]
            self.vmax = extent[-1]
            del_contour = sb_params.get('del_contour')  # making this we can now manage when to delete the contour lines
            # At any part of the thread. Used in GempyModule
            if del_contour:
                self.delete_contourns(ax)
            if self.contours:
                self.add_major_contours(frame, ax, extent[:4])
            if self.minor_contours:
                self.add_minor_contours(frame, ax, extent[:4])
            if self.contours_label and self.contours:
                self.add_label_contours(ax)
        else:
            if self._active:
                self.delete_contourns(ax)
            self._active = active

        return sb_params

    @staticmethod
    def delete_contourns(ax):
        [coll.remove() for coll in reversed(ax.collections) if isinstance(coll, matplotlib.collections.PathCollection)]
        ## this deletes the axes labels as well
        #[text.remove() for text in reversed(ax.texts) if isinstance(text, matplotlib.text.Text)]

    ## dit zijn de contour lijnen
    def plot_contour_lines(self, frame, ax):
        self.add_major_contours(frame, ax)
        self.add_minor_contours(frame, ax)
        self.add_label_contours(ax)

    def add_major_contours(self, data, ax, extent=None):
        """Renders contours to the current plot object.
        Uses the different attributes to style contour lines and contour labels.
        """
        self.major = ax.contour(data,
                                levels=self.contours_levels
                                if len(self.contours_levels) < self._threshold
                                and not self.automatic_contours else None,
                                linewidths=self.contours_width,
                                colors=self.contours_color,
                                extent=extent,
                                zorder=100
                                )

    def add_minor_contours(self, data, ax, extent=None):
        if len(self.contours_levels_minor) < self._threshold and not self.automatic_contours:
            self.minor = ax.contour(data,
                                    levels=self.contours_levels_minor,
                                    linewidths=self.contours_width_minor,
                                    colors=self.contours_color,
                                    extent=extent,
                                    zorder=99
                                    )

    def add_label_contours(self, ax):
        self.label = ax.clabel(self.major,
                               inline=self.contours_label_inline,
                               fontsize=self.contours_label_fontsize,
                               fmt=self.contours_label_format)

    @property
    def contours_levels(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step)

    @property
    def contours_levels_minor(self):
        """Returns the current contour levels, being aware of changes in calibration."""

        return numpy.arange(self.vmin, self.vmax, self.contours_step_minor)

    def show_widgets(self):
        self._create_widgets()
        panel = pn.Column("###<b>Contour lines </b>",
                          self._widget_active_contours,
                          self._widget_automatic_contours,
                          "<b> Modify contour lines manually </b>",
                          pn.WidgetBox(self._widget_plot_contours,
                                       self._widget_plot_step_contours,
                                       self._widget_plot_minorcontours,
                                       self._widget_plot_step_minorcontours,
                                       self._widget_plot_contours_label,
                                       self._widget_plot_contours_label_fontsize
                                       )
                          )

        return panel

    def _create_widgets(self):
        self._widget_active_contours = pn.widgets.Checkbox(name='Active contours', value=self.active_contours)
        self._widget_active_contours.param.watch(self._callback_active_contours, 'value', onlychanged=False)

        self._widget_automatic_contours = pn.widgets.Checkbox(name='Automatic levels for contours',
                                                              value=self.automatic_contours)
        self._widget_automatic_contours.param.watch(self._callback_automatic_contours, 'value', onlychanged=False)

        self._widget_plot_contours = pn.widgets.Checkbox(name='Show major contours', value=self.contours)
        self._widget_plot_contours.param.watch(self._callback_plot_contours, 'value',
                                               onlychanged=False)

        self._widget_plot_minorcontours = pn.widgets.Checkbox(name='Show minor contours', value=self.minor_contours)
        self._widget_plot_minorcontours.param.watch(self._callback_plot_minorcontours, 'value',
                                                    onlychanged=False)
        self._widget_plot_step_contours = pn.widgets.Spinner(name='Choose a contour step', value=self.contours_step)
        self._widget_plot_step_contours.param.watch(self._callback_plot_step_contours, 'value', onlychanged=False)

        self._widget_plot_step_minorcontours = pn.widgets.Spinner(name='Choose a minor contour step',
                                                                  value=self.contours_step_minor)
        self._widget_plot_step_minorcontours.param.watch(self._callback_plot_step_minorcontours, 'value',
                                                         onlychanged=False)

        self._widget_plot_contours_label = pn.widgets.Checkbox(name='Show contours label',
                                                               value=self.contours_label)
        self._widget_plot_contours_label.param.watch(self._callback_plot_contours_label, 'value',
                                                     onlychanged=False)

        self._widget_plot_contours_label_fontsize = pn.widgets.Spinner(name='set a contour label fontsize',
                                                                       value=self.contours_label_fontsize)
        self._widget_plot_contours_label_fontsize.param.watch(self._callback_plot_contours_label_fontsize, 'value',
                                                              onlychanged=False)

    def _callback_automatic_contours(self, event):
        self.automatic_contours = event.new

    def _callback_active_contours(self, event):
        self.active_contours = event.new

    def _callback_plot_contours(self, event):
        self.contours = event.new

    def _callback_plot_minorcontours(self, event):
        self.minor_contours = event.new

    def _callback_plot_step_contours(self, event):
        self.contours_step = event.new

    def _callback_plot_step_minorcontours(self, event):
        self.contours_step_minor = event.new

    def _callback_plot_contours_label(self, event):
        self.contours_label = event.new

    def _callback_plot_contours_label_fontsize(self, event):
        self.contours_label_fontsize = event.new
