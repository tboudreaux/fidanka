import numpy as np

class fiducial_line:
    def __init__(self, name):
        self.name = name
        self._measurements = list()

    def add_measurement(self, color, mag):
        self._measurements.append(np.vstack((color, mag)))

    def __repr__(self):
        return f"<Fiducial Line {self.name} : {len(self._measurements)} measurements>"
