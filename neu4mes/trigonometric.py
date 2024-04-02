import torch
import torch.nn as nn

from neu4mes.relation import Relation, NeuObj
from neu4mes.input import Input
from neu4mes.model import Model

class Sin(Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_sin'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                'Sin':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is Input or
              type(obj) is Relation or
              issubclass(type(obj), Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_sin'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                'Sin':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')



class Cos(Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_cos'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                'Cos':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is Input or
              type(obj) is Relation or
              issubclass(type(obj), Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_cos'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                'Cos':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')



class Tan(Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_tan'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                'Tan':[(obj[0].name,obj[1])],
            }
        elif (type(obj) is Input or
              type(obj) is Relation or
              issubclass(type(obj), Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_tan'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                'Tan':[obj.name]
            }
        else:
            raise Exception('Type is not supported!')


class Sin_Layer(nn.Module):
    def __init__(self,):
        super(Sin_Layer, self).__init__()
    def forward(self, x):
        return torch.sin(x)

def createSin(self, *inputs):
    return Sin_Layer()

class Cos_Layer(nn.Module):
    def __init__(self,):
        super(Cos_Layer, self).__init__()
    def forward(self, x):
        return torch.cos(x)

def createCos(self, *inputs):
    return Cos_Layer()

class Tan_Layer(nn.Module):
    def __init__(self,):
        super(Tan_Layer, self).__init__()
    def forward(self, x):
        return torch.tan(x)

def createTan(self, *inputs):
    return Tan_Layer()


setattr(Model, 'Sin', createSin)
setattr(Model, 'Tan', createTan)
setattr(Model, 'Cos', createCos)