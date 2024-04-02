from neu4mes.relation import NeuObj
from neu4mes.input import Input

class Output(NeuObj):              
    def __init__(self, obj, relation):
        super().__init__(relation.json)
        if type(obj) is tuple:
            self.name = obj[0].name+obj[1]
            self.signal_name = obj[0].name
        elif type(obj) is Input:
            self.name = obj.name
            self.signal_name = obj[0].name
        self.json['Outputs'][self.name] = {}
        if relation.name in self.json['Relations']:
            self.json['Relations'][self.name] = self.json['Relations'][relation.name]
            relations = self.json['Relations']
            
        for key, val in relations[self.name].items():
            for signal in val:
                self.navigateRelations(signal)
    
    def navigateRelations(self,signal):

        if signal in self.json['Relations']:   
            for key, val in self.json['Relations'][signal].items():
                for signal in val:
                    if type(signal) is tuple:
                        self.navigateRelations(signal[0])
                    else:
                        self.navigateRelations(signal)
