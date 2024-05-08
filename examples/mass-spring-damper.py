import sys
import os
# append a new directory to sys.path
sys.path.append(os.getcwd())

from neu4mes import *
from neu4mes.visualizer import StandardVisualizer

# Create neural model
x = Input('x')
F = Input('F')
func_x = Fir(4)
func_f = Fir(4)
x_z = Output(x.z(-1), Fir(func_x(x.tw(2))+func_f(F)+func_x(x.tw(2))))
#y = Output(x.tw(3))
#y = Output('pippo', x.tw(3))
#x_z = Output(func(x.tw(2))+Fir(F)+func(x.tw(2)))
#x_z(x)
#x_z = Minimize(x.z(-1), Fir(x.tw(2))+Fir(F), loss='mse')
#x_z = Minimize(Fir(x.tw(1)), Fir(y.tw(2))+Fir(F), loss='mse')

# Add the neural model to the neu4mes structure and neuralization of the model
mass_spring_damper = Neu4mes(verbose = True,  visualizer = StandardVisualizer())
mass_spring_damper.addModel(x_z)
mass_spring_damper.neuralizeModel(0.05)

from torch.fx import symbolic_trace
model = symbolic_trace(mass_spring_damper.model)
print(model.code)

# Data load
data_struct = ['time','x','x_s','F']
#data_folder = './datasets/mass-spring-damper/data/'
data_folder = os.path.join('examples', 'datasets', 'mass-spring-damper', 'data')
mass_spring_damper.loadData(data_struct, folder = data_folder)

# Neural network train
mass_spring_damper.trainModel(train_size=0.7, show_results = True)

