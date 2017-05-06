import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np


class App(QtGui.QMainWindow):
    def __init__(self):
        super(App, self).__init__()

        # GUI

        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        # 3D

        self.gl_items = {}
        self.canvas3d = gl.GLViewWidget()
        self.mainbox.layout().addWidget(self.canvas3d)
        self.canvas3d.setMinimumHeight(500)

        self.zgrid = gl.GLGridItem()
        self.canvas3d.addItem(self.zgrid)

        # Draw axes
        axes_scale = 10
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]) * axes_scale, color=[1, 0, 0, 1], width=1)
        self.canvas3d.addItem(x_axis)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 1, 0]]) * axes_scale, color=[0, 1, 0, 1], width=1)
        self.canvas3d.addItem(y_axis)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 1]]) * axes_scale, color=[0, 0, 1, 1], width=1)
        self.canvas3d.addItem(z_axis)

        # 2D

        """self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)"""

        self.statusBar().showMessage("Ready")

        # Image
        """self.view = self.canvas.addViewBox(0, 0)
        self.view.setAspectLocked(True)
        self.img = pg.ImageItem(border='w')
        self.img.setImage(img)
        self.view.addItem(self.img)"""

        # Plot
        """self.plot = self.canvas.addPlot(1, 0)
        self.h2 = self.plot.plot(pen='y')"""

        self.showMaximized()
        self.center()

        # Set Data

        """self.x = np.linspace(0, 50., num=100)
        self.X, self.Y = np.meshgrid(self.x, self.x)"""

        self.dt = 0
        self.fps = 0.
        self.then = time.time()

        self.closing_events = []

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        for e in self.closing_events:
            e()
        event.accept()

    def cleanup(self, e):
        # Events to run on quit
        self.closing_events.append(e)

    def update_fps(self):
        fps2 = 1.0 / self.dt
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
        self.statusBar().showMessage(tx)

    def anim(self, func):
        now = time.time()
        self.dt = (now - self.then)
        if self.dt <= 0:
            self.dt = 0.000000000001
        self.then = now

        # self.counter += self.dt * 10
        # self.ydata = np.cos(self.x + self.counter) + np.sin(self.x + self.counter)

        # self.h2.setData(self.ydata)

        # Run code
        func(self.dt)

        self.update_fps()
        QtCore.QTimer.singleShot(1, lambda: self.anim(func))

    def add_gl_item(self, name, item):
        if name not in self.gl_items:
            self.gl_items[name] = item
            self.canvas3d.addItem(item)

    def set_gl_data(self, name, data):
        self.gl_items[name].setData(data)

    def get_gl_item(self, name):
        return self.gl_items[name]


def create_camera(pos):
    verts = np.array([
        [0, 0, 0],
        [1, 0.8, -1.5],
        [-1, 0.8, -1.5],
        [1, -0.8, -1.5],
        [-1, -0.8, -1.5],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 3, 4]
    ])

    camera = gl.GLMeshItem(vertexes=verts, faces=faces, smooth=False, drawEdges=True, drawFaces=False,
                           edgeColor=[0, 0, 0, 1])
    camera.translate(pos[0], pos[1], pos[2])
    return camera


def main():
    app = QtGui.QApplication(sys.argv)
    my_app = App()

    my_app.canvas3d.setBackgroundColor([64, 64, 64])

    camera = create_camera(np.zeros(3))
    my_app.add_gl_item("camera", camera)

    def main_loop(dt):
        pass

    # my_app.zgrid.translate(dt, 0, 0, local=True)
    # print my_app.zgrid.transform()

    my_app.anim(main_loop)

    my_app.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
