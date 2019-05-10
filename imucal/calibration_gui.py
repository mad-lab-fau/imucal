class CalibrationGui:
    section_list = None
    acc_list_markers = None
    gyro_list_markers = None

    def __init__(self, acc, gyro, expected_labels, master=None):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        import tkinter as tk

        self.text_label = 'labels {{}}/{}'.format(len(expected_labels))

        if not master:
            master = tk.Tk()

        # reset variables
        self.section_list = []
        self.acc_list_markers = []
        self.gyro_list_markers = []

        # Create a container
        self.fig, self.axs = self._create_figure(acc, gyro)

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self._onclick)

        self.label_text = tk.Text(master, height=1, width=80)
        self.label_text.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.label_text.insert(tk.END, self.text_label.format(str(0)))

        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        master.mainloop()

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
        return fig, (ax1, ax2)

    def _onclick(self, event):
        import tkinter as tk
        # switch to the move cursor
        # set a marker with doubleclick left
        # remove the last marker with doubleclick right

        # with button 1 (double left click) you will set a marker
        if event.button == 1 and event.dblclick:
            x = int(event.xdata)
            self.section_list.append(x)
            marker_acc = self.axs[0].axvline(x)
            marker_gyro = self.axs[1].axvline(x)
            self.acc_list_markers.append(marker_acc)
            self.gyro_list_markers.append(marker_gyro)

        # with button 3 (double right click) you will remove the last marker
        elif event.button == 3 and event.dblclick:
            # position of the last marker
            x = self.section_list[-1]
            self.acc_list_markers.pop().remove()
            self.gyro_list_markers.pop().remove()
            self.section_list.remove(x)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.label_text.delete('1.0', tk.END)
        self.label_text.insert(tk.END, self.text_label.format(str(len(self.section_list))))
