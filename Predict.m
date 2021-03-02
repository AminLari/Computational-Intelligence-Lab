function [ Out ] = Predict( weight, data )
    [row,column] = size(data);
    Out = zeros(column+1,1);
    for i = 1:row 
            new_data = [data(i,:)'; 1]; 
            Out(i) = .5*sign(weight'*new_data)+.5; 
    end

end

