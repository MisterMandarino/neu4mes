import torch.nn as nn

from neu4mes.relation import Relation, NeuObj
from neu4mes.input import Input
from neu4mes.model import Model

linear_relation_name = 'Linear'
linear_bias_relation_name = 'LinearBias'

class Linear(Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin'+str(NeuObj.count)
            if type(obj[1]) is list:
                if len(obj) == 2:
                    self.json['Relations'][self.name] = {
                        linear_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]))],
                    }
                elif len(obj) == 3:
                    self.json['Relations'][self.name] = {
                        linear_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]),obj[2])],
                    }
                else:
                    raise Exception('Type is not supported!')
            else:
                self.json['Relations'][self.name] = {
                    linear_relation_name:[(obj[0].name,obj[1])],
                }
        elif (type(obj) is Input or
            issubclass(type(obj),Input) or
            type(obj) is Relation or
            issubclass(type(obj), Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_lin'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

class LinearBias(Relation):
    def __init__(self, obj):
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_lin_bias'+str(NeuObj.count)
            if type(obj[1]) is list:
                if len(obj) == 2:
                    self.json['Relations'][self.name] = {
                        linear_bias_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]))],
                    }
                elif len(obj) == 3:
                    self.json['Relations'][self.name] = {
                        linear_bias_relation_name:[(obj[0].name,(obj[1][0],obj[1][1]),obj[2])],
                    }
                else:
                    raise Exception('Type is not supported!')
            else:
                self.json['Relations'][self.name] = {
                    linear_bias_relation_name:[(obj[0].name,obj[1])],
                }
        elif (type(obj) is Input or
            type(obj) is Relation or
            issubclass(type(obj), Relation)):
            super().__init__(obj.json)
            self.name = obj.name+'_lin_bias'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                linear_bias_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

def createLinear(self, input_size):
    return nn.Linear(in_features=input_size, out_features=1, bias=False)

def createLinearBias(self, input_size):
    return nn.Linear(in_features=input_size, out_features=1, bias=True)

setattr(Model, linear_relation_name, createLinear)
setattr(Model, linear_bias_relation_name, createLinearBias)