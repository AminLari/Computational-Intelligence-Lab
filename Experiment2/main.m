clc
clear
data = csvread('breast_cancer.csv',1,0);
%data = fileread('breast_cancer.csv'); %inputs
input =data(:,[1:30]);
desired_out = data(:,31); % expected outputs
weight = learn(input, desired_out, 0.1, 100); % learning weights
Out = Predict(weight,input); % predicting the given data
