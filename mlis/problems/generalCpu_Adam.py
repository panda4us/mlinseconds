
# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.output_size=output_size
        self.loss_function_key=solution.loss_function_key
        self.loss_functions=solution.loss_functions
        #self.activation1=solution.activation_function1
        #self.activation2=solution.activation_function2
        #self.hidden_size1 = solution.hidden_size1
        self.hidden_size1 = 88
        #self.hidden_size2 = solution.hidden_size2
        self.hidden_size2 = 51
        #self.hidden_size3 = solution.hidden_size3
        self.hidden_size3 = self.hidden_size1
        self.linear1 = nn.Linear(input_size, self.hidden_size1)
        self.linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.linear4 = nn.Linear(self.hidden_size3, output_size)
        self.activation1=solution.activations[solution.activation_hidden_key_1]
        self.activation2=solution.activations[solution.activation_hidden_key_2]
        self.activation3=solution.activations[solution.activation_hidden_key_3]
        self.activations=solution.activations
        if solution.batch_norm:
            self.BN1=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
            self.BN2=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
            self.BN3=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
        else:
            self.BN1=lambda x: x
            self.BN2=lambda x: x
            self.BN3=lambda x: x
        
        
        
        
        
        
        
        

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.BN1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.BN2(x)  
        x = self.linear3(x)
        x = self.activation3(x)
        x = self.BN3(x)
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        return self.loss_functions[self.loss_function_key](output,target)
          



    def get_avg_grad(self):
        return [p.grad.abs().mean() for p in self.parameters()]

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 0.007
        # Control number of hidden )neurons
        self.loss_function_key='BCELoss'
        self.hidden_size1 =100
        self.hidden_size2 =44
        self.hidden_size3 =88
        self.hidden_size4 =24
        self.hidden_size5 =8
        
        self.multiplier = 1.25
        self.momentum = 0.8
        self.lr_limit=5
        self.best_correct=0
        self.best_step=200
        self.activation_hidden_key_1 = "hardshrink"
        self.activation_hidden_key_2 = "prelu"
        self.activation_hidden_key_3 = "hardshrink"#"prelu"
        self.batch_norm=True
        
        
        self.activations = {
                'sigmoid': nn.Sigmoid(),
                'relu': nn.ReLU(),
                'relu6': nn.ReLU6(),
                'htang1': nn.Hardtanh(-1, 1),
                'tanh': nn.Tanh(),
                'selu': nn.SELU(),
                'hardshrink': nn.Hardshrink(),
                'prelu': nn.PReLU(),
            }
        self.loss_functions = {
                'BCELoss': nn.BCELoss(),
                'MSELoss': nn.MSELoss(),
                 }
        
        
        
        #self.batch_norm_grid=[True,False]
        #self.activation_hidden_key_1_grid = list(self.activations.keys())
        #self.activation_hidden_key_2_grid = list(self.activations.keys())
        #self.loss_function_key_grid = list(self.loss_functions.keys())
        # Grid search settings, see grid_search_tutorial
        #self.learning_rate_grid =[0.015]
        #self.activation_function=["relu_","selu_","sigmoid"]
        #self.loss_function_grid=["BCELoss","MSE","L1Loss","CrossEntropyLoss","CTCLoss","BCELoss","BCEWithLogitsLoss",]
        #self.loss_function_grid=["MSE","BCELoss"]
        #self.learning_rate_grid =[0.009,0.011]
        #self.activation_function1_grid=["relu_","selu_","sigmoid","tanh"]
        #self.activation_function2_grid=["relu_","selu_","sigmoid","tanh"]
        #self.activation_function1_grid=["relu_"]
        #self.activation_function2_grid=["sigmoid"]
        #self.learning_rate_grid =np.linspace(0.001,0.015,5)
        #self.momentum_grid=np.linspace(0.88,0.98,5)
        self.hidden_size1_grid = [32,64,88]
        #self.hidden_size3_grid = [256,512]
        #self.hidden_size2_grid = [256,512]
        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 5

    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
#        return self.grid_search_tutorial()
        # Model represent our neural network
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # Optimizer used for training neural network
        #sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, betas =(0.5,0.7))
        
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.1)
        while True:
            # Report step, so we know how many steps
            #for g in optimizer.param_groups:
                #print(g['lr'])
                #print(self.multiplier)
                #g['lr'] = g['lr']*self.multiplier
                #if g['lr']>self.lr_limit:
                    #g['lr']=self.learning_rate

            
            
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(train_data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(train_target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1 or correct == total:
                #print("breaking")
                #print(correct)
                #print(total)
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            #print(model.get_avg_grad())
            # print progress of the learning
            #s.write((context.step, error)) 
            self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            #print(error)
            
        #print(context.step,self.grid_search.choice_str)
        if self.grid_search:
            self.grid_search.add_result('step', context.step)
            self.grid_search.add_result('error', error)
            self.grid_search.add_result('correct/total', correct/total)
            
        if self.iter == self.iter_number-1:
            #print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
            #print("[HelloXor] iters={}".format(self.grid_search.get_results('step')))
            stats = self.grid_search.get_stats('step')
            
            if stats[0]<self.best_step:
                print("improved!!, choice={} Mean={:.2f} Std={:.2f}".format(self.grid_search.choice_str,float(stats[0]), float(stats[1])))
                self.best_step=stats[0]
            else: 
                print("not improved!!, choice={} Mean={:.2f} Std={:.2f}".format(self.grid_search.choice_str,float(stats[0]), float(stats[1])))
        return model

    def print_stats(self, step, error, correct, total):
        if step % 1000 == 0:
            print("Step = {} Correct = {}/{} Error = {}".format(step, correct, total, error.item()))
          


###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

run_grid_search = False
# Uncomment next line if you want to run grid search
#run_grid_search = True
if run_grid_search:
    gs.GridSearch().run(Config(), case_number=8, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)
