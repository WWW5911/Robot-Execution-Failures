import numpy as np
import pandas as pd 
import math
import time 

np.random.seed(7)

df_lp1 = 'lp1_data.csv'
df_lp2 = 'lp2_data.csv'
df_lp3 = 'lp3_data.csv'
df_lp4 = 'lp4_data.csv'
df_lp5 = 'lp5_data.csv'
dfs = [df_lp1, df_lp2, df_lp3, df_lp4, df_lp5]

train_ratio = 0.8
learning_rate = 0.03
Epoch = 501
tau = 0.001
Layer = [8, 12, 12, 5]
output_dim = 5

minmax = 1
strr = ""
if minmax == 1:
    strr = 'with minmax normalizion'


full_data = []
train_data = []
lp_tmp = []
for df in dfs:
    tmp = pd.read_csv(df)
    tmp = np.array(tmp)
    for tp in tmp:
        full_data.append(tp)

np.random.shuffle(full_data)
for tp in full_data:
    if tp[6] == 3:
        continue
    train_data.append(tp[0:6])
    if tp[6] == 3:
        lp_tmp.append(2)
    else :
        lp_tmp.append(tp[6])

train_label = np.zeros( ( len(train_data), output_dim ) )
for i in range(len(lp_tmp)):
    train_label[i][lp_tmp[i]-1] = 1     # let 1-5 turns to 0-4

def initialize(Layer):
    Weights = []
    Bias = []
    for i in range ( 1, len(Layer) ):
        #weight = np.random.randn( Layer[i-1], Layer[i]) * np.sqrt(2 / Layer[i])
        #b = np.random.randn(Layer[i], 1) * np.sqrt(2 / Layer[i])
        weight = np.random.rand( Layer[i-1], Layer[i]) * np.sqrt(1 / (Layer[i-1]+ Layer[i]) )
        b = np.zeros( (Layer[i], 1) ) 
        Weights.append(weight)
        Bias.append(b)

    # to make index of layer 1 from 0 to 1
    Weights.insert(0, [])
    Bias.insert(0, [])
    Weights = np.asarray(Weights)
    Bias = np.asarray(Bias)

    return Weights, Bias

def Minmax_scale(w):
    return np.array( (w-w.min()) / (w.max()-w.min()) )

def Partition(data, label, ratio):
    if minmax == 1 :
        data = np.transpose(data).astype(np.float64)
        for i in range (len(data) ):
            data[i] = Minmax_scale(data[i])
        data = np.transpose(data)
    t_data = data[0:int(len(data)*ratio) ]
    t_label = label[0:int(len(label)*ratio) ]
    verify_data = data[int(len(data)*ratio):len(data)]
    verify_label = label[int(len(data)*ratio):len(data)]
    return t_data, t_label, verify_data, verify_label

#print("D")

def Sigmoid( n ):
    l = []
    for tmp in n:
        if np.sum(tmp) >= 0:
            l.append( 1 / (1 + math.exp(-tmp) ) )
        else:
            l.append( math.exp(tmp) / ( 1 + math.exp(tmp) )  )
    return l

# def sigmoid(data):
#   if np.sum(data) >= 0:
#     return  1 / ( 1 + math.exp(-np.sum(data)))
#   else:
#     return math.exp(np.sum(data)) / (1 + math.exp(np.sum(data)) )

def Binary_crossEntropy(y, a):
    CE = 0
    for i in range( len(y) ):
        CE += y[i] * math.log(a[i] + 1e-15 ) + (1 - y[i]) * math.log(1 - a[i] + 1e-15)
    return CE

def FeedForward(data, Weight, bias, active):
    n = np.matmul( data, Weight) + bias.transpose() 
    # return np.asarray( Sigmoid(n[0]) ) 
    return np.asarray( active(n[0]) ) 

def Prediction(a):
    index = 0
    for i in range( len(a) ):
        if a[i] > a[index]:
            index = i
    return index

def softmax( n ):
    n = np.asarray(n)
    t = np.exp(n)/sum(np.exp(n))
    return t

def ReLU( n ):
    n = np.asarray(n)
    for i in range (len(n)):
        if n[i] < 0:
            n[i] = 0
    return n

# add input dim and output dim
Layer.insert(0, len(train_data[0]))
Layer.append(output_dim)

Weights, Bias = initialize(Layer)
t_data, t_label, verify_data, verify_label = Partition(train_data, train_label, train_ratio)



end_flag = False
last_acc_T = 0
last_acc_V = 0
ce = []
writee = False

output_Active = softmax  # in output Layer : softmax / Sigmoid
active = ReLU # in hidden Layer : softmax / Sigmoid / ReLU

for ep in range(Epoch):
    totalLoss = 0
    L = len(Layer)-1
    acc_t = 0
    acc_v = 0
    flag = (ep % 10 == 0)
    string = ""
    for i in range(len(t_data)):
        a = []
        a.append( t_data[i] )

        for l in range(1, L ):
            a.append(FeedForward(a[l-1],  Weights[l], Bias[l], active) )
        
        # output layer 
        n = np.matmul( a[L-1], Weights[L]) + Bias[L].transpose() 
        a.append( output_Active(n[0]) )


        # BackWard
        Delta = [ a[L] - t_label[i] ]
        for l in range(L-1, 0, -1):
            Delta.insert(0, np.matmul( Delta[0], Weights[l+1].transpose() ) * ( a[l]*(1-a[l])  )  )
        Delta.insert(0, [])
        
        # update Weights and Bias
        for l in range(1, L+1):
            t = np.matmul( Delta[l][np.newaxis].T, a[l-1][np.newaxis] )
            Weights[l] = Weights[l] - learning_rate * t.T
            Bias[l] = Bias[l] - learning_rate * Delta[l]
        
        # Calculate loss and training accuracy
        if flag : 
            totalLoss -= Binary_crossEntropy( t_label[i], a[L] )
            p = Prediction(a[L])
            if t_label[i][p] == 1:
                acc_t += 1

    # Calculate verify accuracy
    if flag : 
        for i in range( len(verify_data) ):
            out = verify_data[i]
            for l in range(1, L+1 ):
                out = FeedForward(out, Weights[l], Bias[l], active)
            p = Prediction(out)
            if verify_label[i][p] == 1:
                acc_v += 1

        if last_acc_T < acc_t / len(t_data) and last_acc_V > acc_v / len(verify_data):
            print("< Overtraining leads to overfitting >" )
            end_flag = True
        last_acc_T = acc_t
        last_acc_V = acc_v

    if (ep+1) == Epoch:
        print("< Reach maximum Epoch >" )
        end_flag = True

    if flag and totalLoss / len(t_data) < tau:
        print("< Loss is low enough ( tau: " + str(tau) + ") >"  )
        end_flag = True

    if flag:
        print("<<< Status" + " " + strr + " >>>")
        print("Number of train_data: " + str( len(t_data)) )
        print("Number of verify_data: " + str( len(verify_data)) )
        print("Number of Hidden Layer: " + str( len(Layer)-2) )
        print("Number of Neuron in each Hidden Layer: " + str(Layer[1:len(Layer)-1]) )
        print("Learning Rate: " + str(learning_rate) )
        print("Epoch: " + str(ep+1) )
        print("Loss: " + str( totalLoss / len(t_data) ))
        print("Train accuracy: " + str(acc_t / len(t_data)) )
        print("Verify accuracy: " + str(acc_v / len(verify_data)) )
        
        if writee == True:
            with open('Results\\' + strr + ' ' + 'Result '+time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()) +  '.txt', mode = 'w') as text_file:
                string += "<<<Status>>>\n"
                string += "Number of train_data: " + str( len(t_data)) +"\n"
                string += "Number of verify_data: " + str( len(verify_data)) +"\n"
                string += "Number of Hidden Layer: " + str( len(Layer)-2) +"\n"
                string += "Number of Neuron in each Hidden Layer: " + str(Layer[1:len(Layer)-1]) +"\n"
                string += "Learning Rate: " + str(learning_rate) +"\n"
                string += "Epoch: " + str(ep+1) +"\n"
                string += "Loss: " + str( totalLoss / len(t_data) )+"\n"
                string += "Train accuracy: " + str(acc_t / len(t_data)) +"\n"
                string += "Verify accuracy: " + str(acc_v / len(verify_data)) +"\n"
                text_file.write(string)
                text_file.close()

    if end_flag:
        break
