# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
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
        self.loss_function_key=solution.loss_function_key
        self.loss_functions=solution.loss_functions
        #self.activation1=solution.activation_function1
        #self.activation2=solution.activation_function2
        self.hidden_size1 = solution.hidden_size1
        self.hidden_size2 = solution.hidden_size2
        self.hidden_size3 = solution.hidden_size3
        linear1 = nn.Linear(input_size, self.hidden_size1)
        linear2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        linear3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        linear4 = nn.Linear(self.hidden_size3, output_size)
        activation1=solution.activations[solution.activation_hidden_key_1]
        activation2=solution.activations[solution.activation_hidden_key_2]
        activation3=solution.activations[solution.activation_hidden_key_3]
        activations=solution.activations
        if solution.batch_norm:
            BN1=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
            BN2=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
            BN3=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
            BN4=nn.BatchNorm1d(self.hidden_size1,affine=False,track_running_stats=False)
        else:
            BN1=lambda x: x
            BN2=lambda x: x
            BN3=lambda x: x
            BN4=lambda x: x
        submodels= [linear1,activation1,BN1,linear2,activation2,
        BN2, linear3, activation3, BN3, linear4, nn.Sigmoid()]

        self.sequential= nn.Sequential(*submodels)
        
        
        
        
        
        
        

    def forward(self, x):
       
        return self.sequential(x)

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
        self.learning_rate = 0.07
        # Control number of hidden )neurons
        self.loss_function_key='BCELoss'
        self.hidden_size1 = 64
        self.hidden_size2 = 64
        self.hidden_size3 = 64
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
        self.iter_number = 2

    # Return trained model
    def train_model(self, train_data, train_target, context):
        # Uncommend next line to understand grid search
#        return self.grid_search_tutorial()
        # Model represent our neural network
        
        
        
        #create a dataloader object
        #my_dataset = utils.TensorDataset(train_data,train_target)
        #my_dataloader = utils.DataLoader(my_dataset, batch_size=500, shuffle=True)
        #print('before dataloader')
        #print(train_data.size(),train_target.size())

        #train_data_batch,train_target_baatch= next(iter(my_dataloader))
        #print('after dataloader')
        #print(train_data.size(),train_target.size())

        model = SolutionModel(train_data.size(1), train_target.size(1), self)
        
        # Optimizer used for training neural network
        #sm.SolutionManager.print_hint("Hint[2]: Learning rate is too small", context.step)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer,0.1)
        batch_size= 512
        #print("number of epoch {}".format(train_data.size(0)//batch_size))


        correct_batches= 0    
        while True:
            # Report step, so we know how many steps
            #for g in optimizer.param_groups:
                #print(g['lr'])
                #print(self.multiplier)
                #g['lr'] = g['lr']*self.multiplier
                #if g['lr']>self.lr_limit:
                    #g['lr']=self.learning_rate
            #new batch
            
            #train_data_batch,train_target_batch= next(iter(my_dataloader))
            context.increase_step()
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            ind = random.choice(range(train_data.size(0)//batch_size-1))
            
            #print("starting {} epoch".format(ind))
            data = train_data[batch_size*ind:batch_size*(ind+1)]
            target = train_target[batch_size*ind:batch_size*(ind+1)]
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            # No more time left or learned everything, stop training
            time_left = context.get_timer().get_time_left()
            if correct == total:
                correct_batches=correct_batches+1
                #print("correct batches{}".format(correct_batches))
            if time_left < 0.1 or correct_batches>100: #or correct == total:
                #print("breaking")
                #print(correct)
                #print("finished ".format(ind, correct, total))
                break
            # calculate error
            error = model.calc_error(output, target)
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
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm]
        target = target[perm]
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

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
