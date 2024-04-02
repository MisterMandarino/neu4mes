import torch.nn as nn
import torch

from neu4mes.relation import Relation, merge, NeuObj
from neu4mes.model import Model

sum_relation_name = 'Sum'
minus_relation_name = 'Minus'
subtract_relation_name = 'Subtract'
square_relation_name = 'Square'

class Sum(Relation):
    def __init__(self, obj1, obj2):
        super().__init__(obj1.json)
        self.json = merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sum'+str(NeuObj.count)
        self.json['Relations'][self.name] = {sum_relation_name:[]}
        if type(obj1) is Sum:
            for el in self.json['Relations'][obj1.name]['Sum']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)
        elif type(obj2) is Sum:               
            for el in self.json['Relations'][obj2.name]['Sum']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
        else:
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)

class Subtract(Relation):
    def __init__(self, obj1, obj2):
        super().__init__(obj1.json)
        self.json = merge(obj1.json,obj2.json)
        self.name = obj1.name+'_sub'+str(NeuObj.count)
        self.json['Relations'][self.name] = {sum_relation_name:[]}
        if type(obj1) is Subtract:
            for el in self.json['Relations'][obj1.name]['Sub']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)
        elif type(obj2) is Subtract:               
            for el in self.json['Relations'][obj2.name]['Sub']:
                self.json['Relations'][self.name][sum_relation_name].append(el)
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
        else:
            self.json['Relations'][self.name][sum_relation_name].append(obj1.name)
            self.json['Relations'][self.name][sum_relation_name].append(obj2.name)

class Minus(Relation):
    def __init__(self, obj = None):
        if obj is None:
            return
        super().__init__(obj.json)
        obj_name = obj.name
        self.name = obj.name+'_minus'
        self.json['Relations'][self.name] = {
            minus_relation_name:[obj_name]
        }

class Square(Relation):
    def __init__(self, obj):
        if obj is None:
            return
        super().__init__(obj.json)
        obj_name = obj.name
        self.name = obj.name+'_square'
        self.json['Relations'][self.name] = {
            square_relation_name:[obj_name]
        }

class Minus_Layer(nn.Module):
    def __init__(self):
        super(Minus_Layer, self).__init__()

    def forward(self, x):
        return -x

def createMinus(self, *inputs):
    return Minus_Layer()

class Sum_Layer(nn.Module):
    def __init__(self):
        super(Sum_Layer, self).__init__()

    def forward(self, inputs):
        out = inputs[0]
        for el in inputs[1:]:
            out = out + el
        return out
        #return torch.stack(inputs).sum(dim=0)

def createSum(name, *inputs):
    #return tensorflow.keras.layers.Add(name = name)(input)
    return Sum_Layer()

class Diff_Layer(nn.Module):
    def __init__(self):
        super(Diff_Layer, self).__init__()

    def forward(self, *inputs):
        # Perform element-wise subtraction
        return torch.stack(inputs).diff(dim=0)

def createSubtract(self, *inputs):
    #return tensorflow.keras.layers.Subtract(name = name)(input)
    return Diff_Layer()

class Square_Layer(nn.Module):
    def __init__(self):
        super(Square_Layer, self).__init__()
    def forward(self, x):
        return torch.pow(x,2)

def createSquare(self, *inputs):
    return Square_Layer()

setattr(Model, minus_relation_name, createMinus)
setattr(Model, sum_relation_name, createSum)
setattr(Model, subtract_relation_name, createSubtract)
setattr(Model, square_relation_name, createSquare)