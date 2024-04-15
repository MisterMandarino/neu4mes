
import torch.nn as nn
import torch

from neu4mes.relation import Relation, NeuObj, merge
from neu4mes.model import Model

localmodel_relation_name = 'LocalModel'

class LocalModel(Relation):
    def __init__(self, obj1, obj2):
        self.name = ''
        if type(obj1) is tuple and obj2.values is not None:
            super().__init__(obj1[0].json)
            self.json = merge(obj1[0].json,obj2.json)
            self.name = obj1[0].name+'X'+obj2.name+'_loc'+str(NeuObj.count)
            self.json['Relations'][self.name] = {
                localmodel_relation_name:[(obj1[0].name,obj1[1]),obj2.name],
            }
        else:
            raise Exception('Type is not supported!')

class LocalModel_Layer(nn.Module):
    ## TODO: check if the gradient is propagated only for the active variable
    def __init__(self, input_size, output_size):
        super(LocalModel_Layer, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.classes = output_size
    def forward(self, args):
        linear_out = self.linear(args[0])
        one_hot = torch.nn.functional.one_hot(args[1].to(torch.int64), num_classes=self.classes).squeeze().to(torch.float32)
        out = torch.mul(linear_out, one_hot)
        if len(out.shape) == 1:
            out = torch.sum(out, dim=0, keepdim=True)
        else:
            out = torch.sum(out, dim=1, keepdim=True)

        #one_hot = torch.nn.functional.one_hot(args[1].to(torch.int64), num_classes=self.classes).squeeze().T.to(torch.float32)
        #out = torch.matmul(linear_out, one_hot)
        return out

def createLocalModel(self, input_size, output_size):
    return LocalModel_Layer(input_size, output_size)

setattr(Model, localmodel_relation_name, createLocalModel)