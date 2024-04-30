import copy
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_def, relation_samples):
        super(Model, self).__init__()
        self.inputs = model_def['Inputs']
        self.outputs = model_def['Outputs']
        self.relations = model_def['Relations']
        self.params = model_def['Parameters']
        self.sample_time = model_def['SampleTime']
        self.samples = relation_samples

        ## Build the network
        self.relation_forward = {}
        self.relation_inputs = {}
        for relation, inputs in self.relations.items():
            rel_name = inputs[0]
            func = getattr(self,rel_name)
            if func:
                ## collect the inputs needed for the relation
                input_var = [i[0] if type(i) is tuple else i for i in inputs[1]]

                if rel_name == 'LocalModel': # TODO: Work in progress
                    pass
                    #self.relation_forward[relation] = func(self.n_samples[input_var[0]], len(self.inputs[input_var[1]]['Discrete']))
                    #self.n_samples[relation] = len(self.inputs[input_var[1]]['Discrete'])

                elif rel_name == 'Fir':  ## Linear module requires 2 inputs: input_size and output_size
                    if set(['dim_in', 'dim_out']).issubset(self.params[inputs[2]].keys()):
                        self.relation_forward[relation] = func(self.params[inputs[2]]['dim_in'], self.params[inputs[2]]['dim_out'])
                    elif 'tw_in' in self.params[inputs[2]].keys():
                        if type(self.params[inputs[2]]['tw_in']) is list:  ## Backward + forward
                            dim_in = int(abs(self.params[inputs[2]]['tw_in'][0]) / self.sample_time) + int(abs(self.params[inputs[2]]['tw_in'][1]) / self.sample_time)
                        else:
                            dim_in =  int(self.params[inputs[2]]['tw_in'] / self.sample_time)
                        self.relation_forward[relation] = func(dim_in, self.params[inputs[2]]['dim_out'])
                    elif 'tw_out' in self.params[inputs[2]].keys():
                        if type(self.params[inputs[2]]['tw_out']) is list:  ## Backward + forward
                            dim_out = int(abs(self.params[inputs[2]]['tw_out'][0]) / self.sample_time) + int(abs(self.params[inputs[2]]['tw_out'][1]) / self.sample_time)
                        else:
                            dim_out =  int(self.params[inputs[2]]['tw_out'] / self.sample_time)
                        self.relation_forward[relation] = func(self.params[inputs[2]]['dim_in'], dim_out)

                else: ## Functions that takes no parameters
                    self.relation_forward[relation] = func()
                self.relation_inputs[relation] = input_var       
            else:
                print("Relation not defined")
        self.params = nn.ParameterDict(self.relation_forward)

        #print('[LOG] relation forward: ', self.relation_forward)
        #print('[LOG] relation inputs: ', self.relation_inputs)
    
    def forward(self, kwargs):
        available_inputs = kwargs
        while not set(self.outputs.keys()).issubset(available_inputs.keys()):
            for output in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if (output not in available_inputs.keys()) and (set(self.relation_inputs[output]).issubset(available_inputs.keys())):
                    layer_inputs = [available_inputs[key][:,self.samples[output][key]['backward']:self.samples[output][key]['forward']] for key in self.relation_inputs[output]]
                    if len(layer_inputs) <= 1: ## i have a single forward pass
                        available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                    else:
                        available_inputs[output] = self.relation_forward[output](layer_inputs)

        ## Return a dictionary with all the outputs final values
        result_dict = {key: available_inputs[key] for key in self.outputs.keys()}
        return result_dict
    
    '''
    def forward(self, kwargs):
        #available_inputs = kwargs
        available_inputs = {}
        input_keys = list(self.inputs.keys())
        for key in input_keys:
            available_inputs[key] = kwargs[key]
        while not set(self.outputs.keys()).issubset(input_keys):
            for output in self.relations.keys():
                ## if i have all the variables i can calculate the relation
                if (output not in input_keys) and (set(self.relation_inputs[output]).issubset(input_keys)):
                    layer_inputs = [available_inputs[key] for key in self.relation_inputs[output]]
                    if len(layer_inputs) <= 1: ## i have a single forward pass
                        available_inputs[output] = self.relation_forward[output](layer_inputs[0])
                        input_keys.append(output)
                    else:
                        available_inputs[output] = self.relation_forward[output](layer_inputs)
                        input_keys.append(output)
    '''