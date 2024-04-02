import copy
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, inputs, outputs, relations, n_samples):
        super(Model, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.relations = relations
        self.n_samples = copy.deepcopy(n_samples)

        ## Build the network
        self.relation_forward = {}
        self.relation_inputs = {}
        ## Cycle through all the relations
        for output, relation in self.relations.items():
            ## Cycle through all the variables needed for the relation
            for rel, var in relation.items(): 
                func = getattr(self,rel) ## Get the relation attribute
                if func:
                    # Create the Relation
                    input_var = []
                    for item in var:
                        if type(item) is tuple:
                            input_var.append(item[0])
                        else:
                            input_var.append(item)
                    if rel == 'LocalModel':  ## Needs two inputs: dimension of the filter and the number of discrete variables
                        self.relation_forward[output] = func(self.n_samples[input_var[0]], len(self.inputs[input_var[1]]['Discrete']))
                        self.n_samples[output] = len(self.inputs[input_var[1]]['Discrete'])
                    elif rel == 'Linear' or rel == 'LinearBias': ## Needs one input: dimension of the input
                        self.relation_forward[output] = func(self.n_samples[input_var[0]])
                        self.n_samples[output] = 1
                    else: ## No inputs needed
                        self.relation_forward[output] = func()
                        self.n_samples[output] = self.n_samples[input_var[0]]

                    ## Save the list of inputs needed to obtain the relation
                    self.relation_inputs[output] = input_var
                else:
                    print("Relation not defined")

        ## Use the parameterDict in order to have a gradient for the optimizer
        self.params = nn.ParameterDict(self.relation_forward)

    def forward(self, kwargs):
        available_inputs = kwargs
        while not set(self.outputs.keys()).issubset(available_inputs.keys()):
            for output in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if (output not in available_inputs.keys()) and (set(self.relation_inputs[output]).issubset(available_inputs.keys())):
                    layer_inputs = [available_inputs[key] for key in self.relation_inputs[output]]
                    if len(layer_inputs) <= 1: ## i have a single forward pass
                        available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                    else:
                        available_inputs[output] = self.relation_forward[output](layer_inputs)

        ## Return a dictionary with all the outputs final values
        result_dict = {key: available_inputs[key] for key in self.outputs.keys()}
        return result_dict