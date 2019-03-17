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

dataset = df.head(150).values.tolist()

training_dataset = dataset[:40]+dataset[50:90]+dataset[100:140]
validasi_dataset = dataset[40:50]+dataset[90:100]+dataset[140:]

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

theta_for_hidden = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
theta_for_output = [0.45,0.55,0.65,0.75]
bias1 = 0.5
bias2 = 0.6
alpha = 0.1
hidden_result = [0.0,0.0]
activation_hidden = [0.0,0.0]
output_result = [0.0,0.0]
prediction=[0.0,0.0]
prediksi_validasi=[0.0,0.0]
output_derivative = [0.0,0.0,0.0,0.0]
theta_out_hidd_for_derivative = [0.0,0.0,0.0,0.0]
hidden_derivative = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
n_epoch = 500
accuracy_training = []
error_training = []
accuracy_validasi = []
error_validasi = []

for n in range(n_epoch):
        sum_error = 0.0
        ctr = 0
        ctr_validasi = 0
        sum_error_validasi = 0.0
        #print(theta_for_hidden)
#training
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

                output_derivative[0] = derivative_out(training_dataset[j][4],act(output_result[0]),hidden_result[0])
                output_derivative[1] = derivative_out(training_dataset[j][4],act(output_result[0]),hidden_result[1])
                output_derivative[2] = derivative_out(training_dataset[j][5],act(output_result[1]),hidden_result[0])
                output_derivative[3] = derivative_out(training_dataset[j][5],act(output_result[1]),hidden_result[1])

                for i in range(len(theta_for_output)):
                        theta_for_output[i] = update(theta_for_output[i],alpha,output_derivative[i])

                theta_out_hidd_for_derivative[0]=theta_for_output[0]
                theta_out_hidd_for_derivative[1]=theta_for_output[2]
                theta_out_hidd_for_derivative[2]=theta_for_output[1]
                theta_out_hidd_for_derivative[3]=theta_for_output[3]

                for k in range(len(hidden_derivative)):
                        for m in range(len(hidden_result)):
                                for i in range(4):
                                        if m == 0:
                                                hidden_derivative[k]=derivative_hidd(training_dataset[j][4:6],output_result,theta_out_hidd_for_derivative[:2],act(hidden_result[m]),training_dataset[j][i])
                                        else :
                                                hidden_derivative[k]=derivative_hidd(training_dataset[j][4:6],output_result,theta_out_hidd_for_derivative[2:],act(hidden_result[m]),training_dataset[j][i])
                                 
                for i in range(len(theta_for_hidden)):
                        theta_for_hidden[i] = update(theta_for_hidden[i],alpha,hidden_derivative[i])

                sum_error += error_total
                
        error_training.append(sum_error / len(training_dataset))
        accuracy_training.append(ctr/len(training_dataset))
#validasi
        for z in range(len(validasi_dataset)):
                error_total = 0.0
#forward pass
                for i in range(len(hidden_result)):
                        if i==0:
                                hidden_result[i] = hasil(validasi_dataset[z],theta_for_hidden[:4],bias1)
                                activation_hidden[i]=act(hidden_result[i])
                        else:
                                hidden_result[i] = hasil(validasi_dataset[z],theta_for_hidden[4:],bias1)
                                activation_hidden[i]=act(hidden_result[i])
        
                for i in range(len(output_result)):
                        if i==0:
                                output_result[i] = hasil(activation_hidden,theta_for_output[:2],bias2)
                        else:
                                output_result[i] = hasil(activation_hidden,theta_for_output[2:],bias2)

                for i in range(len(output_result)):
                        error_total += err(validasi_dataset[z][i+4],act(output_result[i]))
                        if act(output_result[i]) >= 0.5:
                                prediction[i] = 1.0
                        else: prediction[i] = 0.0
                        
                if (validasi_dataset[z][4] == prediction[0] and validasi_dataset[z][5] == prediction[1]):
                        ctr_validasi = ctr_validasi + 1

                sum_error_validasi += error_total
        
        error_validasi.append(sum_error_validasi / len(validasi_dataset))
        accuracy_validasi.append(ctr_validasi / len(validasi_dataset))

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

