function [ Weight ] = learn(data, pre, learning_cnst, epoches ) 
 
    [row, column] = size(data); 
    Weight = zeros(column + 1, 1);
    for j= 1:epoches 
        for i = 1:row  
            new_data = [data(i,:)'; 1]; 
            out = 0.5*sign(Weight'*new_data)+0.5;
            Weight = Weight - learning_cnst * (out - pre(i)) * new_data;
        end 
    end 
end


