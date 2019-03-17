import matplotlib.pyplot as plt
import pandas as pd
import math

#referensi

#A Step by Step Backpropagation Example
#https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

idx = ['x1','x2','x3','x4','types']
df = pd.read_csv("iris.csv",names=idx)
df['fakta1'] = 1
df['fakta2'] = 1
df.loc[df['types'] == 'setosa','fakta1'] = 0
df.loc[df['types'] == 'setosa','fakta2'] = 0
df.loc[df['types'] == 'versicolor','fakta1'] = 0
df.loc[df['types'] == 'virginica','fakta2'] = 0
df = df.drop(['types'], axis = 1)

training = []
training.append(pd.concat([df.iloc[:40], df.iloc[50:90], df.iloc[100:140]]))

training_dataset = df.head(120).values.tolist()

validasi = []
validasi.append(pd.concat([df.iloc[40:50], df.iloc[90:100], df.iloc[140:]]))
validasi_dataset = df.head(30).values.tolist()

def hasil(row, theta, bias):
        hasil = bias
        for i in range(len(theta)):
                hasil += theta[i] * float(row[i])
        return hasil

def act(hasil):
        activation = 1/(1+math.exp(-hasil))
        return activation

def err(fakta, activation):
        error = (1/2) * math.pow(activation-fakta,2)
        return error

def derivative_out(fakta, out_result, hidd_result):
        y1 = out_result-fakta
        y2 = out_result*(1-out_result)
        return y1*y2*hidd_result

def update(thetaORbias, alpha, derivative):
        new_thetaORbias = thetaORbias - alpha * derivative
        return new_thetaORbias

def derivative_hidd(fakta, out_result, theta_for_out_each_hidd, hidd_result, x):
        sum_bla = 0.0
        for i in range(len(fakta)):
                new_something = (act(out_result[i])-fakta[i])*act(out_result[i])*(1-act(out_result[i]))*theta_for_out_each_hidd[i]
                sum_bla += new_something
        return sum_bla * hidd_result * (1-hidd_result) * x

def valid(train, hiddenTheta, outputTheta, bias1, bias2):
        prediksi = [0.0,0.0]
        keluaran = [0.0,0.0]
        sum_errorvalid = 0.0
        counter = 0
        result_hidd = [0.0,0.0]
        result_out = [0.0,0.0]
        hidd_act = [0.0,0.0]
        for j in range(len(train)):
                error_tot = 0.0
#forward pass
                for i in range(len(result_hidd)):
                        if i==0:
                                result_hidd[i] = hasil(train[j],hiddenTheta[:4],bias1)
                                hidd_act[i] = act(result_hidd[i])
                        else:
                                result_hidd[i] = hasil(train[j],hiddenTheta[4:],bias1)
                                hidd_act[i] = act(result_hidd[i])
        
                for i in range(len(result_out)):
                        if i==0:
                                result_out[i] = hasil(hidd_act,outputTheta[:2],bias2)
                        else:
                                result_out[i] = hasil(hidd_act,outputTheta[2:],bias2)

                for i in range(len(result_out)):
                        error_tot += err(training_dataset[j][i+4],act(result_out[i]))
                        if act(result_out[i]) >= 0.5:
                                prediksi[i] = 1.0
                        else: prediksi[i] = 0.0

                if (training_dataset[j][4] == prediction[0] and training_dataset[j][5] == prediction[1]):
                        counter+=1
                sum_errorvalid += error_tot
        keluaran[0] = sum_errorvalid / len(train)
        keluaran[1] = counter/len(train)
        return keluaran
        

theta_for_hidden = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
theta_for_output = [0.45,0.55,0.65,0.75]
bias1 = 0.5
bias2 = 0.6
alpha = 0.1
hidden_result = [0.0,0.0]
activation_hidden = [0.0,0.0]
output_result = [0.0,0.0]
prediction=[0.0,0.0]
n_epoch = 5
accuracy_training = []
error_training = []
accuracy_validasi = []
error_validasi = []

for n in range(n_epoch):
        sum_error = 0.0
        ctr = 0
        output_derivative = []
        theta_out_hidd_for_derivative = []
        hidden_derivative = []
        for j in range(len(training_dataset)):
                error_total = 0.0
#forward pass
                for i in range(len(hidden_result)):
                        if i==0:
                                hidden_result[i] = hasil(training_dataset[j],theta_for_hidden[:4],bias1)
                                activation_hidden[i]=act(hidden_result[i])
                        else:
                                hidden_result[i] = hasil(training_dataset[j],theta_for_hidden[4:],bias1)
                                activation_hidden[i]=act(hidden_result[i])
        
                for i in range(len(output_result)):
                        if i==0:
                                output_result[i] = hasil(activation_hidden,theta_for_output[:2],bias2)
                        else:
                                output_result[i] = hasil(activation_hidden,theta_for_output[2:],bias2)

#backpropagation
                for i in range(len(output_result)):
                        error_total += err(training_dataset[j][i+4],act(output_result[i]))
                        if act(output_result[i]) >= 0.5:
                                prediction[i] = 1.0
                        else: prediction[i] = 0.0
                if (training_dataset[j][4] == prediction[0] and training_dataset[j][5] == prediction[1]):
                        ctr = ctr + 1

                for i in range(len(output_result)):
                        for j in range(len(hidden_result)):
                                output_derivative.append(derivative_out(training_dataset[j][i+4],act(output_result[i]),hidden_result[j]))

                for i in range(len(theta_for_output)):
                        theta_for_output[i] = update(theta_for_output[i],alpha,output_derivative[i])

                theta_out_hidd_for_derivative.append(theta_for_output[0])
                theta_out_hidd_for_derivative.append(theta_for_output[2])
                theta_out_hidd_for_derivative.append(theta_for_output[1])
                theta_out_hidd_for_derivative.append(theta_for_output[3])

                for j in range(len(hidden_result)):
                        for i in range(4):
                                if j == 0:
                                        hidden_derivative.append(derivative_hidd(training_dataset[j][4:6],output_result,theta_out_hidd_for_derivative[:2],act(hidden_result[j]),training_dataset[j][i]))
                                else :
                                        hidden_derivative.append(derivative_hidd(training_dataset[j][4:6],output_result,theta_out_hidd_for_derivative[2:],act(hidden_result[j]),training_dataset[j][i]))
                                 
                for i in range(len(theta_for_hidden)):
                        theta_for_hidden[i] = update(theta_for_hidden[i],alpha,hidden_derivative[i])

                #print(error_total)
                sum_error += error_total
        error_training.append(sum_error / len(training_dataset))
        accuracy_training.append(ctr/len(training_dataset))
        
        x = valid(validasi_dataset, theta_for_hidden, theta_for_output, bias1, bias2)
        error_validasi.append(x[0])
        accuracy_validasi.append(x[1])
        
plt.figure('Accuracy (alpha = 0.1)')
plt.plot(accuracy_training,'b-', label='training')
plt.plot(accuracy_validasi,'r-', label='validasi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')

plt.figure('Error (alpha = 0.1)')
plt.plot(error_training,'b-', label='training')
plt.plot(error_validasi,'r-', label='validasi')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.show()

