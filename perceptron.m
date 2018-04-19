clear all;
clc;

load MNIST_digit_data.mat
range = 1:500;
rate = 0.1;
number_of_iterations=1000;
%total = 1000;
y = labels_train;

%rand_thousand = randperm(total);
[y_rows,~]=size(y);
for target_change=1:y_rows
    if(y(target_change) == 6)
        y(target_change)=1;
    elseif(y(target_change) == 1)
        y(target_change)=0;
    else
        y(target_change)=-1;
    end
end

range_6_index = find(y>0);
range_1_index = find(y==0);

X_6 = images_train(range_6_index,:);
y_6 = y(range_6_index,:);
X_6 = X_6(range,:);
y_6 = y_6(range,:);

X_1 = images_train(range_1_index,:);
y_1 = y(range_1_index,:);
X_1 = X_1(range,:);
y_1 = y_1(range,:);

X = vertcat(X_1,X_6);
y = vertcat(y_1,y_6);

[x_rows,x_cols]= size(X);
weight_matrix = zeros(1,1+x_cols);
[~,w_cols]=size(weight_matrix);
for i=1:number_of_iterations
    for x_tar=1:x_rows
        target = y(x_tar);
        xi = X(x_tar,:);
        update = rate * (target - prediction(xi,weight_matrix));
        %if(target ~= predict(xi,weight_matrix))
            weight_matrix(:,2:w_cols) = weight_matrix(:,2:w_cols) + update * xi;
            weight_matrix(:,1) = weight_matrix(:,1) + update;
        %end
    end
end

X_test = images_test;
y_test = labels_test;

[y_rows_test,~]=size(y_test);

for test_change=1:y_rows_test
    if(y_test(test_change) == 6)
        y_test(test_change)=1;
    elseif(y_test(test_change)== 1)
        y_test(test_change)=0;
    else 
         y_test(test_change)=-1;
    end
end

range_6_index_test = find(y_test>0);
range_1_index_test = find(y_test==0);

X_6_test = X_test(range_6_index_test,:);
y_6_test = y_test(range_6_index_test,:);
X_6_test = X_6_test(range,:);
y_6_test = y_6_test(range,:);

X_1_test = X_test(range_1_index_test,:);
y_1_test = y_test(range_1_index_test,:);
X_1_test = X_1_test(range,:);
y_1_test = y_1_test(range,:);

X_test = vertcat(X_1_test,X_6_test);
y_test = vertcat(y_1_test,y_6_test);

[y_rows_test,~]=size(y_test);
correct_predict = 0;
for i=1:y_rows_test
    temp(i,1) = prediction(X_test(i,:),weight_matrix);
    if(temp(i,1)== y_test(i,1))
        correct_predict = correct_predict + 1;
    end
end
accuracy = correct_predict/y_rows_test;
accuracy_percentage = accuracy * 100;
fprintf('Accuracy for %d iterations = %2.4f\n',number_of_iterations,accuracy_percentage);