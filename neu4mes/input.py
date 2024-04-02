import numpy as np

from neu4mes.relation import NeuObj

class Input(NeuObj):
    def __init__(self,name,values = None):
        super().__init__()
        self.name = name
        self.json['Inputs'][self.name] = {}
        if values:
            self.values = values
            self.json['Inputs'][self.name] = {
                'Discrete' : values
            }

    def tw(self, tw, offset = None):
        if offset is not None:
            return self, tw, offset
        return self, tw

    def z(self, advance):
        if advance > 0:
            return self, '__+z'+str(advance)
        else:
            return self, '__-z'+str(-advance)

    def s(self, derivate):
        if derivate > 0:
            return self, '__+s'+str(derivate)
        else:
            return self, '__-s'+str(-derivate)

class ControlInput(Input):
    def __init__(self,name,values = None):
        super().__init__(name,values)
