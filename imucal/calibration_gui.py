from collections import OrderedDict
from itertools import chain

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk


class CalibrationGui:
    section_list = None
    acc_list_markers = None
    gyro_list_markers = None
    expected_labels = None

    def __init__(self, acc, gyro, expected_labels, master=None):
        self.expected_labels = expected_labels
        cmap = matplotlib.cm.get_cmap('Set3')
        self.colors = {k: matplotlib.colors.to_hex(cmap(i/12)) for i, k in enumerate(expected_labels)}

        self.text_label = 'missing_labels {{}}/{}'.format(len(expected_labels) * 2)

        if not master:
            master = tk.Tk()

        # reset variables
        self.section_list = OrderedDict(((k, [None, None]) for k in expected_labels))
        self.acc_list_markers = {k: [] for k in expected_labels}
        self.gyro_list_markers = {k: [] for k in expected_labels}

        self.main_area = tk.Frame(master=master, relief=tk.RAISED)
        self.main_area.pack(fill=tk.BOTH, expand=1)

        self.side_bar = tk.Frame(master=self.main_area)
        self.side_bar.pack(fill=tk.Y, side=tk.RIGHT)
        self._create_sidebar()

        # Create a container
        self.fig, self.axs = self._create_figure(acc, gyro)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self._onclick)

        self.label_text = tk.Text(master, height=1, width=80)
        self.label_text.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.label_text.insert(tk.END, self.text_label.format(str(0)))

        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()

        master.mainloop()

    def _create_sidebar(self):
        self.missing_labels = tk.Listbox(master=self.side_bar, width=30)
        self.missing_labels.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.existing_labels = tk.Listbox(master=self.side_bar)
        self.existing_labels.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.existing_labels.bind('<Double-1>', lambda x: self._exisiting_label_clicked())

        self._update_list_boxes()

    def _exisiting_label_clicked(self):
        selected_key = self.existing_labels.get(tk.ACTIVE)
        self.section_list[selected_key] = [None, None]
        self._update_list_boxes()
        self._update_marker(selected_key)

    def _update_list_boxes(self):
        self.existing_labels.delete(0, tk.END)
        for item in [k for k, v in self.section_list.items() if all(v)]:
            self.existing_labels.insert(tk.END, item)
            self.existing_labels.itemconfig(self.existing_labels.size() - 1, {'bg': self.colors[item]})

        old_active = self.missing_labels.get(tk.ACTIVE)
        self.missing_labels.delete(0, tk.END)
        new_missing = [k for k, v in self.section_list.items() if not all(v)]
        for item in new_missing:
            self.missing_labels.insert(tk.END, item)
            self.missing_labels.itemconfig(self.missing_labels.size() - 1, {'bg': self.colors[item]})

        if old_active not in self.missing_labels.get(0, tk.END):
            self.missing_labels.activate(0)
        else:
            self.missing_labels.activate(new_missing.index(old_active))

    def _create_figure(self, acc, gyro):
        from matplotlib.figure import Figure

        fig = Figure(figsize=(20, 10))
        ax1 = fig.add_subplot(211)
        ax1.plot(acc)
        ax1.grid(True)
        ax1.set_title("Set a label at start/end of accelerometer placements (12 in total)")
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("acceleration [m/s^2]")
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(gyro)
        ax2.grid(True)
        ax2.set_title("Set a label at start/end of gyroscope rotation (6 in total)")
        ax2.set_xlabel("time[s]")
        ax2.set_ylabel("rotation [°/s]")
        fig.tight_layout()
        return fig, (ax1, ax2)

    def _onclick(self, event):
        if event.button not in [1, 3]:
            return

        selected_key = self.missing_labels.get(tk.ACTIVE)

        if event.button == 1:
            x = int(event.xdata)
            self.section_list[selected_key][0] = x
        elif event.button == 3:
            x = int(event.xdata)
            self.section_list[selected_key][1] = x

        if all(self.section_list[selected_key]):
            self.section_list[selected_key].sort()

        self._update_marker(selected_key)

        self.label_text.delete('1.0', tk.END)
        self.label_text.insert(tk.END, self.text_label.format(str(len(self.section_list))))

        self._update_list_boxes()

    def _update_marker(self, key):

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
            a_acc = self.axs[0].axvspan(self.section_list[key][0], self.section_list[key][1], alpha=0.5, color=self.colors[key])
            a_gyr = self.axs[1].axvspan(self.section_list[key][0], self.section_list[key][1], alpha=0.5, color=self.colors[key])
            self.acc_list_markers[key].append(a_acc)
            self.gyro_list_markers[key].append(a_gyr)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

