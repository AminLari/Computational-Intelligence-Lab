clear 
clc

input = [0 0;0 1;1 0;1 1];
output = [0; 1; 1; 0];
learning_rate = 0.1;
epoches = 10000;

input_layer_neurons = 2;
hidden_layer_neurons = 2;
output_layer_neurons = 1;


weight_first_layer = random('Normal',0,1,input_layer_neurons, hidden_layer_neurons) ;
bias_first_layer = random('Normal',0,1,1, hidden_layer_neurons);
weight_second_layer = random('Normal',0,1,hidden_layer_neurons, output_layer_neurons);
bias_second_layer = random('Normal',0,1,1, output_layer_neurons);

    for i = 1:epoches
        [hidden_layer_output,~] = sigmoid(input * weight_first_layer + bias_first_layer); 
        [network_output,~] = sigmoid(hidden_layer_output * weight_second_layer + bias_second_layer);
        
        Error = 0.5 * (output - network_output);
        delta_network=Error.* (1- network_output.*network_output); 
        delta_hidden = 0.5 * (1 - hidden_layer_output.*hidden_layer_output).* (delta_network * transpose(weight_second_layer)); 
        
  
        if all( delta_network > .01) && all( delta_hidden > .01)
            break
        end
        
        weight_second_layer = weight_second_layer + learning_rate*(transpose(hidden_layer_output) *delta_network); 
        weight_first_layer = weight_first_layer + learning_rate*(transpose(input) *delta_hidden);
        bias_second_layer = bias_second_layer +  sum(delta_network) * learning_rate; 
        bias_first_layer = bias_first_layer +  sum(delta_hidden) * learning_rate;
    end
   
network_output

function [f,fdot] = sigmoid(x)
f = (2./((1+exp(-x))))-1;
fdot = ((1-(f.^2))./2);
end