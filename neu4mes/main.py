import torch
from torch.utils.data import DataLoader

import numpy as np
import os
from pprint import pprint
import re
from datetime import datetime
import matplotlib.pyplot as plt

from neu4mes.relation import NeuObj, merge
from neu4mes.visualizer import TextVisualizer
from neu4mes.dataset import Neu4MesDataset
from neu4mes.loss import CustomRMSE
from neu4mes.output import Output
from neu4mes.model import Model

class Neu4mes:
    def __init__(self, model_def = 0, verbose = False, visualizer = TextVisualizer()):
        # Set verbose print inside the class
        self.verbose = verbose

        # Inizialize the model definition
        # the model_def has all the relation defined for that model
        if type(model_def) is Output:
            self.model_def = model_def.json
        elif type(model_def) is dict:
            self.model_def = self.model_def
        else:
            self.model_def = NeuObj().json

        # Input, output, and model characteristics
        self.input_tw_backward = {}         # dimensions of the time window in the past for each input
        self.input_tw_forward = {}          # dimensions of the time window in the future for each input
        self.max_samples_backward = 0       # maxmimum number of samples backward for all the inputs
        self.max_samples_forward = 0        # maxmimum number of samples forward for all the inputs
        self.max_n_samples = 0              # maxmimum number of samples for all the inputs
        self.input_ns_backward = {}         # maxmimum number of samples backward for each the input
        self.input_ns_forward = {}          # maxmimum number of samples forward for each the input
        self.input_n_samples = {}           # maxmimum number of samples for each the input

        self.inputs = {}                    # NN element - processed network inputs
        self.rnn_inputs = {}                # RNN element - processed network inputs
        self.inputs_for_model = {}          # NN element - clean network inputs
        self.rnn_inputs_for_model = {}      # RNN element - clean network inputs
        self.rnn_init_state = {}            # RNN element - for the states of RNN
        self.relations = {}                 # NN element - operations
        self.outputs = {}                   # NN element - clean network outputs

        self.output_relation = {}           # dict with the outputs
        self.output_keys = []               # clear output signal keys (without delay info string __-z1)

        # Models of the framework
        self.model = None                   # NN model - Pytorch model
        self.rnn_model = None               # RNN model - Pytorch model
        self.net_weights = None             # NN weights

        # Optimizer parameters
        self.optimizer = None                     # NN model - Pytorch optimizer
        self.loss_fn = None                 # RNN model - Pytorch loss function

        # Dataset characteristics
        self.input_data = {}                # dict with data divided by file and symbol key: input_data[(file,key)]
        self.inout_data_time_window = {}    # dict with data divided by signal ready for network input
        self.rnn_inout_data_time_window = {}# dict with data divided by signal ready for RNN network input
        self.inout_asarray = {}             # dict for network input in asarray format
        self.rnn_inout_asarray = {}         # dict for RNN network input in asarray format
        self.num_of_samples = None          # number of rows of the file
        self.num_of_training_sample = 0     # number of rows for training

        self.idx_of_rows = [0]              # Index identifying each file start
        self.first_idx_test = 0             # Index identifying the first test

        # Dataloaders
        self.train_loader = None
        self.test_loader = None

        # Training params
        self.batch_size = 128                               # batch size
        self.learning_rate = 0.0005                         # learning rate for NN
        self.num_of_epochs = 20                             # number of epochs
        self.rnn_batch_size = self.batch_size               # batch size for RNN
        self.rnn_window = None                              # window of the RNN
        self.rnn_learning_rate = self.learning_rate/10000   # learning rate for RNN
        self.rnn_num_of_epochs = 50                         # number of epochs for RNN

        # Training dataset
        self.inout_4train = {}                              # Dataset for training NN
        self.inout_4test = {}                               # Dataset for test NN
        self.rnn_inout_4train = {}                          # Dataset for training RNN
        self.rnn_inout_4test = {}                           # Dataset for test RNN

        # Training performance
        self.performance = {}                               # Dict with performance parameters for NN

        # Visualizer
        self.visualizer = visualizer                        # Class for visualizing data

    def MP(self,fun,arg):
        if self.verbose:
            fun(arg)

    """
    Add a new model to be trained:
    :param model_def: can be a json model definition or a Output object
    """
    def addModel(self, model_def):
        if type(model_def) is Output:
            self.model_def = merge(self.model_def, model_def.json)
        elif type(model_def) is dict:
            self.model_def = merge(self.model_def, model_def)
        self.MP(pprint,self.model_def)

    """
    Definition of the network structure through the dependency graph and sampling time.
    If a prediction window is also specified, it means that a recurrent network is also to be defined.
    :param sample_time: the variable defines the rate of the network based on the training set
    :param prediction_window: the variable defines the prediction horizon in the future
    """
    def neuralizeModel(self, sample_time = 0, prediction_window = None):
        # Prediction window is used for the recurrent network
        if prediction_window is not None:
            self.rnn_window = round(prediction_window/sample_time)
            assert prediction_window >= sample_time

        # Sample time is used to define the number of sample for each time window
        if sample_time:
            self.model_def["SampleTime"] = sample_time

        # Look for all the inputs referred to each outputs
        # Set the maximum time window for each input
        relations = self.model_def['Relations']
        for outel in self.model_def['Outputs']:
            relel = relations.get(outel)
            for reltype, relvalue in relel.items():
                self.__setInput(relvalue, outel)

        self.MP(pprint,{"window_backward": self.input_tw_backward, "window_forward":self.input_tw_forward})

        # Building the inputs considering the dimension of the maximum time window forward + backward
        for key,val in self.model_def['Inputs'].items():
            input_ns_backward_aux = int(self.input_tw_backward[key]/self.model_def['SampleTime'])
            input_ns_forward_aux = int(-self.input_tw_forward[key]/self.model_def['SampleTime'])

            # Find the biggest window backwards for building the dataset
            if input_ns_backward_aux > self.max_samples_backward:
                self.max_samples_backward = input_ns_backward_aux

            # Find the biggest horizon forwars for building the dataset
            if input_ns_forward_aux > self.max_samples_forward:
                self.max_samples_forward = input_ns_forward_aux

            # Find the biggest n sample for building the dataset
            if input_ns_forward_aux+input_ns_backward_aux > self.max_n_samples:
                self.max_n_samples = input_ns_forward_aux+input_ns_backward_aux

            # Defining the number of sample for each signal
            if self.input_n_samples.get(key):
                if input_ns_backward_aux+input_ns_forward_aux > self.input_n_samples[key]:
                    self.input_n_samples[key] = input_ns_backward_aux+input_ns_forward_aux
                if input_ns_backward_aux > self.input_ns_backward[key]:
                    self.input_ns_backward[key] = input_ns_backward_aux
                if input_ns_forward_aux > self.input_ns_forward[key]:
                    self.input_ns_forward[key] = input_ns_forward_aux
            else:
                self.input_n_samples[key] = input_ns_backward_aux+input_ns_forward_aux
                self.input_ns_backward[key] = input_ns_backward_aux
                self.input_ns_forward[key] = input_ns_forward_aux

        self.MP(print,"max_n_samples:"+str(self.max_n_samples))
        self.MP(pprint,{"input_n_samples":self.input_n_samples})
        self.MP(pprint,{"input_ns_backward":self.input_ns_backward})
        self.MP(pprint,{"input_ns_forward":self.input_ns_forward})

        ## Build the network
        self.model = Model(self.model_def['Inputs'],self.model_def['Outputs'],self.model_def['Relations'], self.input_n_samples)

    #
    # Recursive method that terminates all inputs that result in a specific relationship for an output
    # During the procedure the dimension of the time window for each input is define
    #
    def __setInput(self, relvalue, outel):
        for el in relvalue:
            if type(el) is tuple:
                if el[0] in self.model_def['Inputs']:
                    time_window = self.input_tw_backward.get(el[0])
                    if time_window is not None:
                        if type(el[1]) is tuple:
                            if self.input_tw_backward[el[0]] < el[1][0]:
                                self.input_tw_backward[el[0]] = el[1][0]
                            if self.input_tw_forward[el[0]] > el[1][1]:
                                self.input_tw_forward[el[0]] = el[1][1]
                        else:
                            if self.input_tw_backward[el[0]] < el[1]:
                                self.input_tw_backward[el[0]] = el[1]
                    else:
                        if type(el[1]) is tuple:
                            self.input_tw_backward[el[0]] = el[1][0]
                            self.input_tw_forward[el[0]] = el[1][1]
                        else:
                            self.input_tw_backward[el[0]] = el[1]
                            self.input_tw_forward[el[0]] = 0
                else:
                    raise Exception("A window on internal signal is not supported!")
            else:
                if el in self.model_def['Inputs']:
                    time_window = self.input_tw_backward.get(el)
                    if time_window is None:
                        self.input_tw_backward[el] = self.model_def['SampleTime']
                        self.input_tw_forward[el] = 0
                else:
                    relel = self.model_def['Relations'].get((outel,el))
                    if relel is None:
                        relel = self.model_def['Relations'].get(el)
                        if relel is None:
                            raise Exception("Graph is not completed!")
                    for reltype, relvalue in relel.items():
                        self.__setInput(relvalue, outel)

    """
    Loading of the data set files and generate the structure for the training considering the structure of the input and the output
    :param format: it is a list of the variable in the csv. All the input keys must be inside this list.
    :param folder: folder of the dataset. Each file is a simulation.
    :param sample_time: number of lines to be skipped (header lines)
    :param delimiters: it is a list of the symbols used between the element of the file
    """
    def loadData(self, format, folder = './data', skiplines = 0, delimiters=['\t',';',',']):
        path, dirs, files = next(os.walk(folder))
        file_count = len(files)

        self.MP(print, "Total number of files: {}".format(file_count))

        # Create a vector of all the signals in the file + output_relation keys
        output_keys = self.model_def['Outputs'].keys()
        for key in format+list(output_keys):
            self.inout_data_time_window[key] = []

        # Read each file
        for file in files:
            for data in format:
                self.input_data[(file,data)] = []

            # Open the file and read lines
            with open(os.path.join(folder,file), 'r') as all_lines:
                lines = all_lines.readlines()[skiplines:] # skip first lines to avoid NaNs

                # Append the data to the input_data dict
                for line in range(0, len(lines)):
                    delimiter_string = '|'.join(delimiters)
                    splitline = re.split(delimiter_string,lines[line].rstrip("\n"))
                    for idx, key in enumerate(format):
                        try:
                            self.input_data[(file,key)].append(float(splitline[idx]))
                        except ValueError:
                            self.input_data[(file,key)].append(splitline[idx])

                # Add one sample if input look at least one forward
                add_sample_forward = 0
                if self.max_samples_forward > 0:
                    add_sample_forward = 1

                # Create inout_data_time_window dict
                # it is a dict of signals. Each signal is a list of vector the dimensions of the vector are (tokens, input_n_samples[key])
                if 'time' in format:
                    for i in range(0, len(self.input_data[(file,'time')])-self.max_n_samples+add_sample_forward):
                        self.inout_data_time_window['time'].append(self.input_data[(file,'time')][i+self.max_n_samples-1-self.max_samples_forward])

                for key in self.input_n_samples.keys():
                    for i in range(0, len(self.input_data[(file,key)])-self.max_n_samples+add_sample_forward):
                        aux_ind = i+self.max_n_samples+self.input_ns_forward[key]-self.max_samples_forward
                        if self.input_n_samples[key] == 1:
                            self.inout_data_time_window[key].append(self.input_data[(file,key)][aux_ind-1])
                        else:
                            self.inout_data_time_window[key].append(self.input_data[(file,key)][aux_ind-self.input_n_samples[key]:aux_ind])

                for key in output_keys:
                    used_key = key
                    elem_key = key.split('__')
                    if len(elem_key) > 1 and elem_key[1]== '-z1':
                        used_key = elem_key[0]
                    for i in range(0, len(self.input_data[(file,used_key)])-self.max_n_samples+add_sample_forward):
                        self.inout_data_time_window[key].append(self.input_data[(file,used_key)][i+self.max_n_samples-self.max_samples_forward])

                # Index identifying each file start
                self.idx_of_rows.append(len(self.inout_data_time_window[list(self.input_n_samples.keys())[0]]))

        # Build the asarray for numpy
        for key,data in self.inout_data_time_window.items():
            self.inout_asarray[key]  = np.asarray(data)

    #
    # Function that get specific parameters for training
    #
    def __getTrainParams(self, training_params):
        if bool(training_params):
            self.batch_size = (training_params['batch_size'] if 'batch_size' in training_params else self.batch_size)
            self.learning_rate = (training_params['learning_rate'] if 'learning_rate' in training_params else self.learning_rate)
            self.num_of_epochs = (training_params['num_of_epochs'] if 'num_of_epochs' in training_params else self.num_of_epochs)
            self.rnn_batch_size = (training_params['rnn_batch_size'] if 'rnn_batch_size' in training_params else self.rnn_batch_size)
            self.rnn_learning_rate = (training_params['rnn_learning_rate'] if 'rnn_learning_rate' in training_params else self.rnn_learning_rate)
            self.rnn_num_of_epochs = (training_params['rnn_num_of_epochs'] if 'rnn_num_of_epochs' in training_params else self.rnn_num_of_epochs)

    """
    Analysis of the results
    """
    def resultAnalysis(self, train_loss, test_loss, test_data):
        ## Plot train loss and test loss
        plt.plot(train_loss, label='train loss')
        plt.plot(test_loss, label='test loss')
        plt.legend()
        plt.show()

        # List of keys
        output_keys = list(self.model_def['Outputs'].keys())

        # Performance parameters
        self.performance['se'] = np.empty([len(output_keys),len(test_data)])
        self.performance['mse'] = np.empty([len(output_keys),])
        self.performance['rmse_test'] = np.empty([len(output_keys),])
        self.performance['fvu'] = np.empty([len(output_keys),])

        # Prediction on test samples
        test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, shuffle=False)
        self.prediction = np.empty((len(output_keys), len(test_data)))
        self.label = np.empty((len(output_keys), len(test_data)))

        with torch.inference_mode():
            self.model.eval()
            for idx, (X, Y) in enumerate(test_loader):
                pred = self.model(X)
                for i, key in enumerate(output_keys):
                    self.prediction[i][idx] = pred[key].item() 
                    self.label[i][idx] = Y[key].item() 
                    self.performance['se'][i][idx] = np.square(pred[key].item() - Y[key].item())

            for i, key in enumerate(output_keys):
                # Mean Square Error 
                self.performance['mse'][i] = np.mean(self.performance['se'][i])
                # Root Mean Square Error
                self.performance['rmse_test'][i] = np.sqrt(np.mean(self.performance['se'][i]))
                # Fraction of variance unexplained (FVU) 
                self.performance['fvu'][i] = np.var(self.prediction[i] - self.label[i]) / np.var(self.label[i])

            # Index of worst results
            self.performance['max_se_idxs'] = np.argmax(self.performance['se'], axis=1)

            # Akaike’s Information Criterion (AIC) test
            ## TODO: Use log likelihood instead of MSE
            #self.performance['aic'] = - (self.num_of_test_sample * np.log(self.performance['mse'])) + 2 * self.model.count_params()

        self.visualizer.showResults(self, output_keys, performance = self.performance)

    """
    Training of the model.
    :param states: it is a list of a states, the state must be an Output object
    :param training_params: dict that contains the parameters of training (batch_size, learning rate, etc..)
    :param test_percentage: numeric value from 0 to 100, it is the part of the dataset used for validate the performance of the network
    :param show_results: it is a boolean for enable the plot of the results
    """
    def trainModel(self, train_size=0.7, training_params = {}, show_results = False):

        # Check input
        self.__getTrainParams(training_params)

        ## Split train and test
        X_train, Y_train = {}, {}
        X_test, Y_test = {}, {}
        for key,data in self.inout_data_time_window.items():
            if data:
                samples = np.asarray(data)
                if samples.ndim == 1:
                    samples = np.reshape(samples, (-1, 1))

                if key in self.model_def['Inputs'].keys():
                    X_train[key] = samples[:int(len(samples)*train_size)]
                    X_test[key] = samples[int(len(samples)*train_size):]
                elif key in self.model_def['Outputs'].keys():
                    Y_train[key] = samples[:int(len(samples)*train_size)]
                    Y_test[key] = samples[int(len(samples)*train_size):]

        ## Build the dataset
        train_data = Neu4MesDataset(X_train, Y_train)
        test_data = Neu4MesDataset(X_test, Y_test)
        

        self.train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)
        self.test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

        ## define optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = CustomRMSE()

        train_losses, test_losses = np.zeros(self.num_of_epochs), np.zeros(self.num_of_epochs)

        for iter in range(self.num_of_epochs):
            self.model.train()
            start = datetime.now()
            train_loss = []
            for X, Y in self.train_loader:
                #inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                out = self.model(X)
                loss = self.loss_fn(out, Y)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
            train_loss = np.mean(train_loss)

            self.model.eval()
            test_loss = []
            for X, Y in self.test_loader:
                #inputs, labels = inputs.to(device), labels.to(device)
                out = self.model(X)
                loss = self.loss_fn(out, Y)
                test_loss.append(loss.item())
            test_loss = np.mean(test_loss)

            train_losses[iter] = train_loss
            test_losses[iter] = test_loss

            if iter % 10 == 0:
                time = datetime.now() - start
                print(f'Epoch {iter+1}/{self.num_of_epochs}, Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, Duration: {time}')

        # Show the analysis of the Result
        if show_results:
            self.resultAnalysis(train_losses, test_losses, test_data)