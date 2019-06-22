# HelloXor is a HelloWorld of Machine Learning.
import time
import random
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils import gridsearch as gs

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        #sm.SolutionManager.print_hint("Hint[1]: Increase hidden size")
        self.hidden_size = solution.hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu_(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

    def calc_error(self, output, target):
        # This is loss function
        return ((output-target)**2).sum()

    def calc_predict(self, output):
        # Simple round output to predict value
        return output.round()

class Solution():
    def __init__(self):
        # Control speed of learning
        self.learning_rate = 1.15
        # Control number of hidden neurons
        self.hidden_size =20
        self.multiplier = 1
        self.momentum = 0.965
        self.lr_limit=15

        # Grid search settings, see grid_search_tutorial
        #self.learning_rate_grid =np.linspace(1,1.3,3)
        self.momentum_grid=np.linspace(0.95,0.98,3)
        #self.hidden_size_grid = [18,20,22]
        # grid search will initialize this field
        self.grid_search = None
        # grid search will initialize this field
        self.iter = 0
        # This fields indicate how many times to run with same arguments
        self.iter_number = 100

    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
#        return self.grid_search_tutorial()
        # Model represent our neural network
        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        # Optimizer used for training neural network
        #sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.1)
        while True:
            # Report step, so we know how many steps
            #for g in optimizer.param_groups:
                #print(g['lr'])
                #print(self.multiplier)
                #g['lr'] = g['lr']*self.multiplier
                #if g['lr']>self.lr_limit:
                #    g['lr']=self.learning_rate

            
            
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
                break
            # calculate error
            error = model.calc_error(output, train_target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            error.backward()
            # print progress of the learning
            self.print_stats(context.step, error, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            #print(error)
            
        #print(context.step,self.grid_search.choice_str)
        if self.grid_search:
            self.grid_search.add_result('step', context.step)
        if self.iter == self.iter_number-1:
            #print("[HelloXor] chose_str={}".format(self.grid_search.choice_str))
            #print("[HelloXor] iters={}".format(self.grid_search.get_results('step')))
            stats = self.grid_search.get_stats('step')
            print("lr={} Mean={:.2f} Std={:.2f}".format(self.grid_search.choice_str,float(stats[0]), float(stats[1])))
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
        self.size_limit = 100
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self):
        data = torch.FloatTensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
            ])
        target = torch.FloatTensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
            ])
        return (data, target)

    def create_case_data(self, case):
        data, target = self.create_data()
        return sm.CaseData(case, Limits(), (data, target), (data, target))

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
    gs.GridSearch().run(Config(), case_number=1, random_order=False, verbose=False)
else:
    # If you want to run specific case, put number here
    sm.SolutionManager().run(Config(), case_number=-1)