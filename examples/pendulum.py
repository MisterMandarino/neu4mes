import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neu4mes structure
pendolum = Neu4mes(verbose = True, visualizer = StandardVisualizer())

# Create neural model
theta = Input('theta')
T     = Input('torque')
lin_theta = Fir(theta.tw(1.5))
sin_theta = Fir(Sin(theta))
torque = Fir(T)
theta_z = Output(theta.z(-1), lin_theta+sin_theta+torque)

# Add the neural model to the neu4mes structure and neuralization of the model
pendolum.addModel(theta_z)
pendolum.neuralizeModel(0.05)

from torch.fx import symbolic_trace
model = symbolic_trace(pendolum.model)
print(model.code)

# Data load
data_struct = ['time','theta','theta_s','','','torque']
#data_folder = './datasets/pendulum/data/'
data_folder = os.path.join('examples', 'datasets', 'pendulum', 'data')
pendolum.loadData(data_struct, folder = data_folder)

# Neural network train
pendolum.trainModel(train_size=0.7, show_results = True)