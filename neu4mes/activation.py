import torch.nn as nn

from neu4mes.relation import Relation
from neu4mes.input import Input
from neu4mes.model import Model

relu_relation_name = 'ReLU'

class Relu(Relation):
    def __init__(self, obj = None):
        if obj is None:
            return
        self.name = ''
        if type(obj) is tuple:
            super().__init__(obj[0].json)
            self.name = obj[0].name+'_relu'
            self.json['Relations'][self.name] = {
                relu_relation_name:[(obj[0].name,obj[1])],
            }
        elif type(obj) is Input:
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                relu_relation_name:[obj.name]
            }
        elif issubclass(type(obj),Relation):
            super().__init__(obj.json)
            self.name = obj.name+'_relu'
            self.json['Relations'][self.name] = {
                relu_relation_name:[obj.name]
            }
        else:
            raise Exception('Type is not supported!')

    def createElem(self, name, input):
        return 


def createRelu(self, *input):
    return nn.ReLU()

setattr(Model, relu_relation_name, createRelu)