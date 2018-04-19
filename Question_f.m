clear all;
clc;
 
load MNIST_digit_data.mat
split = 1:500;        %splitting in half for 2 labels
n=1000;
number_of_iterations =1;

X = images_train;
y = labels_train;
 
X_test = images_test;
y_test = labels_test;
[am,ru]=size(y_test);
error = floor(0.1 * ru);  
rand_10 = randperm(error)';
for pri =1:error
    if(y(rand_10(pri)==1))
        y(rand_10(pri)) = 6;
    elseif(y(rand_10(pri)==6))
        y(rand_10(pri)) = 1;
    end
end
%----- Convert the Labels for 1--> -1 and 6 --> 1(Training and Test)
%----------Training------------
[rows_y,~]=size(y);
for i= 1:rows_y
    if(y(i) ==6)
        y(i) = 1;
    elseif(y(i)==1)
        y(i) = -1;
    else
        y(i) = 0;
    end
end
%test
[rows_y_test,~]=size(y_test);
for i= 1:rows_y_test
    if(y_test(i) ==6)
        y_test(i) = 1;
    elseif(y_test(i)==1)
        y_test(i) = -1;
    else
        y_test(i) = 0;
    end
end
 
indexsix = find(y>0);
indexone = find(y<0);
 
X_new_train_6 = X(indexsix(split),:);
y_new_train_6 = y(indexsix(split),:);
X_new_train_1 = X(indexone(split),:);
y_new_train_1 = y(indexone(split),:);
 
X_new_train = vertcat(X_new_train_1,X_new_train_6);
y_new_train = vertcat(y_new_train_1,y_new_train_6);
 
%------Training ends--------------
 
%----------Test Starts--------
 
indexsix_test = find(y_test>0);
indexone_test = find(y_test<0);
 
X_new_test_6 = X_test(indexsix_test(split),:);
y_new_test_6 = y_test(indexsix_test(split),:);
X_new_test_1 = X_test(indexone_test(split),:);
y_new_test_1 = y_test(indexone_test(split),:);
 
X_new_test = vertcat(X_new_test_1,X_new_test_6);
y_new_test = vertcat(y_new_test_1,y_new_test_6);
 
 
 
%---Pick random data---------------
rand('seed',1);
random_data = randperm(n)';
 
%----- Initialize Weights and Bias to Zero------
[x_train_rows,x_train_cols]=size(X_new_train);
[y_train_rows,y_train_cols]=size(y_new_train);

[x_test_rows,x_test_cols]=size(X_new_test);
[y_test_rows,y_test_cols]=size(y_new_test);

w = zeros(1,x_train_cols);
b=0;

%----- Perceptron Training Model---------------
 
for j=1:number_of_iterations
    for o =1:x_train_rows
        a = dot(w(1,:),X_new_train(random_data(o),:))+ b;
        if(y_new_train(random_data(o),1)*a<=0)
            w=w+y_new_train(random_data(o),1)*X_new_train(random_data(o),:);
            b = b + y_new_train(random_data(o),1);
        end
    end
end
 
 
%----- Perceptron Testing Model------------------
correct =0;
for i=1:1000
    a_test = predict(X_new_test(i,:),w,b);
    if(y_new_test(i) == a_test)
        correct = correct + 1;
    end
end
accuracy = (correct/1000)*100;
fprintf('Accuracy for 1000 iterations is %2.4f\n',accuracy);