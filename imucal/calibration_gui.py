"""Helper providing a small GUI to label timeseries data."""

import tkinter as tk
from collections import OrderedDict
from collections.abc import Sequence
from itertools import chain
from tkinter.messagebox import showinfo
from typing import Optional

import numpy as np


class CalibrationGui:
    """A Gui that can be used to label the different required sections of a calibration.

    For details see the `manual` string.

    Examples
    --------
    >>> # This will launch the GUI and block execution until closed
    >>> gui = CalibrationGui(acc, gyro, ["label1", "label2"])
    >>> # While the GUI is open the sections are labeled. Once closed the next line is executed
    >>> labels = gui.section_list
    >>> labels["label1"]  # First value is start, second value is end of region
    (12, 355)
    >>> labels["label2"]  # First value is start, second value is end of region
    (500, 758)

    Notes
    -----
    If the toolbar is not visible at the bottom of the GUI, try reducing the `initial_figsize` parameter, when
    creating the GUI.

    """

    section_list = None
    acc_list_markers = None
    gyro_list_markers = None
    expected_labels = None

    manual = """
    Mark the start and end point of each section listed in the sidebar in the plots.
    The section that is currently labeled is marked in blue in the sidebar.
    You can use either <return> (<Enter>) or <space> to advance to the next section, which has missing labels
    or click a section label manual.

    To create a mark click either with the left or the right mouse button on the plot.
    For each section you can place one RightClick and one LeftClick label.
    It does not matter, which you place first.
    If both labels are placed, the region in between them is colored.
    Now you can either press Enter (or click on any other label in the sidebar) to continue with labeling the next
    section or you can adjust the labels by repeated left and right clicks until you're satisfied.

    Tip: You can use the matplotlib toolbar to zoom in and out or to pan the plots.
    For faster operations, you can toggle the matplotlib tools using there respective shortcuts:

    Zoom: 'o'
    Pan: 'p'
    Home: 'h'
    Back: 'b'
    Forward: 'f'

    After zooming in, you can use the mouse wheel to scroll along the x-axis.
    """

    def __init__(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        expected_labels: Sequence[str],
        title: Optional[str] = None,
        master: Optional[tk.Frame] = None,
        initial_figsize=(20, 10),
    ) -> None:
        """Launch new GUI instance.

        Parameters
        ----------
        acc :
            3D array containing all acceleration data
        gyro :
            3D array containing all gyroscope data
        expected_labels :
            List of all label names that should be labeled
        title :
            Title displayed in the titlebar of the GUI
        master :
            Parent window if GUI should be embedded in larger application

        """
        import matplotlib
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )

        class CustomToolbar(NavigationToolbar2Tk):
            def __init__(self, canvas, window) -> None:
                # List of buttons to remove
                buttons_to_remove = ["Subplots", "Save"]
                print(self.toolitems)
                self.toolitems = [item for item in self.toolitems if item[0] not in buttons_to_remove]
                super().__init__(canvas, window)

        self.expected_labels = expected_labels
        cmap = matplotlib.cm.get_cmap("Set3")
        self.colors = {k: matplotlib.colors.to_hex(cmap(i / 12)) for i, k in enumerate(expected_labels)}

        self.text_label = f"Labels set: {{}}/{len(expected_labels)}"

        if not master:
            master = tk.Tk()

        master.title(title or "Calibration Gui")
        master.bind("<Return>", lambda _: self._select_next(self.labels.curselection()[0]))
        master.bind("<space>", lambda _: self._select_next(self.labels.curselection()[0]))

        # reset variables
        self.section_list = OrderedDict((k, [None, None]) for k in expected_labels)
        self.acc_list_markers = {k: [] for k in expected_labels}
        self.gyro_list_markers = {k: [] for k in expected_labels}

        self.main_area = tk.Frame(master=master, relief=tk.RAISED)
        self.main_area.pack(fill=tk.BOTH, expand=1)

        self.side_bar = tk.Frame(master=self.main_area)
        self.side_bar.pack(fill=tk.Y, side=tk.RIGHT)
        self._create_sidebar()

        # Create a container
        self.fig, self.axs = _create_figure(acc, gyro, figsize=initial_figsize)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect("button_press_event", self._onclick)

        toolbar = CustomToolbar(self.canvas, self.main_area)
        toolbar.update()

        # To make the shortcut works, we need to manually forward the key event to the toolbar
        # Bind keypress events to the canvas widget
        self.canvas.get_tk_widget().bind("<p>", lambda _: toolbar.pan())
        self.canvas.get_tk_widget().bind("<o>", lambda _: toolbar.zoom())
        self.canvas.get_tk_widget().bind("<h>", lambda _: toolbar.home())
        self.canvas.get_tk_widget().bind("<b>", lambda _: toolbar.back())
        self.canvas.get_tk_widget().bind("<f>", lambda _: toolbar.forward())

        # Bind mouse wheel event to scroll along the x-axis
        self.canvas.get_tk_widget().bind("<MouseWheel>", self._scroll_x)
        self.canvas.get_tk_widget().bind("<Button-4>", self._scroll_x)  # For Linux
        self.canvas.get_tk_widget().bind("<Button-5>", self._scroll_x)  # For Linux

        self.label_text = tk.Text(self.main_area, height=1, width=80, state=tk.DISABLED)
        self.label_text.pack(side=tk.TOP, fill=tk.X, expand=0)
        self._update_text_label(self.text_label.format(str(0)))

        self.labels.selection_anchor(0)
        self.labels.selection_set(0)

        toolbar.pack(side=tk.TOP, fill=tk.X, expand=0)

        # This way, we can start labeling right away
        self.labels.selection_set(0)

        master.mainloop()

    def _scroll_x(self, event) -> None:
        ax = self.canvas.figure.gca()
        x_min, x_max = ax.get_xlim()
        data_min, data_max = ax.dataLim.intervalx
        delta = x_max - x_min
        scroll_in_units = delta * 0.1  # Adjust the scroll speed as needed

        if data_min >= x_min and data_max <= x_max:
            # No scrolling, when we see all the data
            return

        # But we allow scrolling past the ends a little
        data_min -= delta * 0.1
        data_max += delta * 0.1

        if event.num == 5 or event.delta < 0:  # Scroll down -> right
            new_x_max = min(x_max + scroll_in_units, data_max)
            new_x_min = new_x_max - delta
        elif event.num == 4 or event.delta > 0:  # Scroll up -> left
            new_x_min = max(x_min - scroll_in_units, data_min)
            new_x_max = new_x_min + delta

        ax.set_xlim(new_x_min, new_x_max)
        self.canvas.draw_idle()

    def _create_sidebar(self) -> None:
        self.labels = tk.Listbox(master=self.side_bar, width=30)
        self.labels.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        help_button = tk.Button(master=self.side_bar, height=2, text="Help", command=self._show_help)
        help_button.pack(side=tk.BOTTOM, fill=tk.BOTH)

        for key in self.section_list:
            self.labels.insert(tk.END, key)

        self._update_list_box()

    def _show_help(self) -> None:
        showinfo("Window", self.manual)

    def _select_next(self, current) -> None:
        next_val = (current + 1) % self.labels.size()

        if all(list(self.section_list.values())[next_val]):
            if self._n_labels() == len(self.section_list):
                return

            self._select_next(next_val)

        self.labels.selection_clear(0, tk.END)
        self.labels.selection_anchor(next_val)
        self.labels.selection_set(next_val)

    def _update_list_box(self) -> None:
        for i, (key, val) in enumerate(self.section_list.items()):
            self.labels.itemconfig(i, {"bg": self.colors[key]})
            if all(val):
                self.labels.itemconfig(i, {"fg": "#a5a9af"})

    def _update_text_label(self, new_text) -> None:
        self.label_text.config(state=tk.NORMAL)
        self.label_text.delete("1.0", tk.END)
        self.label_text.insert(tk.END, new_text)
        self.label_text.config(state=tk.DISABLED)

    def _onclick(self, event) -> None:
        # Only listen to left and right mouse clicks and only if we are not in zoom or drag mode.
        if event.button not in [1, 3] or str(self.canvas.toolbar.mode):
            return

        selected_index = self.labels.curselection()

        if not selected_index:
            return

        selected_key = self.labels.get(selected_index)

        if event.button == 1:
            x = int(event.xdata)
            self.section_list[selected_key][0] = x
        elif event.button == 3:
            x = int(event.xdata)
            self.section_list[selected_key][1] = x

        if all(self.section_list[selected_key]):
            self.section_list[selected_key].sort()

        self._update_marker(selected_key)

        self._update_text_label(self.text_label.format(str(self._n_labels())))

        self._update_list_box()

    def _update_marker(self, key) -> None:
        for line in chain(self.acc_list_markers[key], self.gyro_list_markers[key]):
            line.remove()
            self.acc_list_markers[key] = []
            self.gyro_list_markers[key] = []

        for val in self.section_list[key]:
            if val:
                marker_acc = self.axs[0].axvline(val, c=self.colors[key])
                marker_gyro = self.axs[1].axvline(val, c=self.colors[key])
                self.acc_list_markers[key].append(marker_acc)
                self.gyro_list_markers[key].append(marker_gyro)

        if all(self.section_list[key]):
            a_acc = self.axs[0].axvspan(
                self.section_list[key][0], self.section_list[key][1], alpha=0.5, color=self.colors[key]
            )
            a_gyr = self.axs[1].axvspan(
                self.section_list[key][0], self.section_list[key][1], alpha=0.5, color=self.colors[key]
            )
            self.acc_list_markers[key].append(a_acc)
            self.gyro_list_markers[key].append(a_gyr)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _n_labels(self):
        return sum(all(v) for v in self.section_list.values())


def _create_figure(acc, gyro, figsize=(20, 10)):
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    lines = ax1.plot(acc)
    ax1.grid(True)
    ax1.set_title("Use this plot to find the static regions for acc calibration")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("acceleration [m/s^2]")
    ax1.legend(lines, list("xyz"))
    ax2 = fig.add_subplot(212, sharex=ax1)
    lines = ax2.plot(gyro)
    ax2.grid(True)
    ax2.set_title("Use this plot to find the single axis rotations for gyro calibration")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("angular velocity [°/s]")
    ax2.legend(lines, list("xyz"))
    fig.tight_layout()
    return fig, (ax1, ax2)
