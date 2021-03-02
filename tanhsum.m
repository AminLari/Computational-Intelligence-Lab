function sum = tanhsum()
    sum = 0;
    col = input('enter number of columns: ');
    row = input('enter number of rows: ');
    matrix = input('enter matrix elements:');
    
    for i = 1:col*row
        sum = sum + tanh(matrix(i));
    end
end

% To test the program, we use this matrix:
% M = [1,0,sin(pi/4);0,1,sin(pi/2);1,0,1]