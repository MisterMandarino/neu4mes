import Neu4mes

mymodel = Neu4mes.Neu4mes()
model_def = {
    'SampleTime':0.05,
    'Input':{
        'x1':{
            'Name':'Mass 1 position'
        },
        'F':{
            'Name':'External force'
        }
    },
    'Output':{
        'x1_z':{     #con z indico il ritardo unitario di una variabile     
            'Name':'Next mass 1 position'
        }
    },
    'Relations':{
        'x1_z':{
            'Linear':[('x1',2),'F'],
        }
    }
}

# model_def = {
#     'SampleTime':0.05,
#     'Input':{
#         'x1':{
#             'Name':'Mass 1 position'
#         },
#         'F':{
#             'Name':'External force'
#         }
#     },
#     'Output':{
#         'x1_z':{     #con z indico il ritardo unitario di una variabile     
#             'Name':'Next mass 1 position'
#         },
#         'x1_s':{    #con s indico la defivata di un segnale
#             'Name':'Velocity of mass 1'
#         }
#     },
#     'Relations':{
#         'x1_z':{
#             'Linear':[('x1',2),'F'],
#         },
#         'x1_s':{
#             'Linear':[('x1',3),'F'],
#         },
#     }
# }



# massamolla = Neu4mes.Neu4mes()
# x1 = Input('x1')
# F = Input('Force')
# x1_z = Output('x1_z', x1.z(1), Linear(x1.tw(2))+Linear(F))
# massamolla.modelDefinition(x1_z)
# massamolla.neuralizeModel()

# gear = DiscreteInput('gear',dimension = 8)
# engine = Input('engine')
# brake = Input('brake')
# altitude = Input('altitude')
# velocity = Input('velocity')
# acceleration = Input('accleration')

# x1_z = Output('x1_s2', acceleration, NonNegativeLinear(brake.tw(1.25))+Linear(velocity^2)+Linear(altitude.tw(2))+LocalModel(engine.tw(2),gear))
mymodel.modelDefinition(model_def)
mymodel.neuralizeModel()

# data = {
#     'time' : [[],[]]
#     'x1' : [[simulazione 1],[simulazione 2]]
#     'F'  : [[],[]]
# }
data_struct = ['time','x1','x1_s','F']
data_folder = './data/data-linear-oscillator-a/'
mymodel.loadData(data_struct, folder = data_folder)
mymodel.trainModel(validation_percentage = 30)
#mymodel.showResults()


#Examples:
#1. Vehicle model + control system
#2. State estimator for lateral velocity
#3. Signle/double Mass-spring-dumper
#4. Cart-Pole
#5. Pedestrian estrimator
# model_def = {
#     'SampleTime':0.05,
#     'Input':{
#         'x1':{
#             'Name':'Mass 1 position'
#         },
#         'F':{
#             'Name':'External force'
#         }
#     },
#     'Output':{
#         'x1_z':{     #con z indico il ritardo unitario di una variabile     
#             'Name':'Next mass 1 position'
#         },
#         'x1_s':{    #con s indico la defivata di un segnale
#             'Name':'Velocity of mass 1'
#         }
#     },
#     'Relations':{
#         'x1_z':{
#             'Linear':[('x1',2),'F'],
#         },
#         'x1_s':{
#             'Linear':[('x1',3),'F'],
#         },
#     }
# }
# model_def = {
#     'SampleTime':0.05,
#     'Input':{
#         'x1':{
#             'Name':'Mass 1 position'
#         },
#         'x2':{
#             'Name':'Mass 2 position'
#         },
#         'F':{
#             'Name':'External force'
#         },
#         'T':{
#             'Name':'External tension'
#         }
#     },
#     'Output':{
#         'x1p':{
#             'Name':'Next mass 1 position'
#         },
#         'x2p':{
#             'Name':'Next mass 2 position'
#         }
#     },
#     'Relations':{
#         'x1p':{
#             'Linear':[('x1',2),'F'],
#         },
#         'x2p':{
#             'Linear':[('x2',2),('F',1),'T'],
#         },
#     }
# }

# model_def = {
#     'Input':{
#         'omega':{},
#         'u':{},
#         'ay':{},
#         'ax':{}
#     },
#     'Output':{
#         'v':{}
#     },
#     'Params':{
#         'gamma':{},
#         'beta':{}
#     },
#     'State':{
#         'local_u':{},
#         'loval_v':{}
#     },
#     'Relations':{
#         'local_v':{
#             'Function':{
#                'eq': lambda v, ay, u, omega: mymodel.gamma*v+mymodel.beta*mymodel.Ts*(ay-v*omega)
#             }
#         },
#         'local_u':{
#             'Function':{
#                'eq': lambda v, ay, u, omega: mymodel.gamma*v+mymodel.beta*mymodel.Ts*(ay-v*omega)
#             }
#         },
#         'v':{
#             'LocalModel':[('local_v'),'omega']
#         }
#     }
# }