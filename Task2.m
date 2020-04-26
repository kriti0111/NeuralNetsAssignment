%Task 2 - Test
load discriminant_variables.mat
load test.mat

std_test_data = standardisation(test_data);
[M,N] = size(test_data);
[x,y] = size(train_data);
disp('Starting Task 2');

%uncomemnt based on the kernel type n 

%accuracy for hard-margin linear kernel
% [predicted_label_test, accuracy_test] = get_accuracy_linear(std_test_data,opt_W1,opt_B1,N,test_label);
% [predicted_label_train, accuracy_train] = get_accuracy_linear(std_train_data,opt_W1,opt_B1,y,train_label);

%accuracy for hard-margin Polynomial kernel
[predicted_label_test, accuracy_test] = get_accuracy_poly(std_test_data,opt_W2,opt_B2,N,test_label);
[predicted_label_train, accuracy_train] = get_accuracy_poly(std_train_data,opt_W2,opt_B2,y,train_label);

%accuracy for soft-margin Polynomial kernel
% [predicted_label_test, accuracy_test] = get_accuracy_poly(std_test_data,opt_W3,opt_B3,N,test_label);
% [predicted_label_train, accuracy_train] = get_accuracy_poly(std_train_data,opt_W3,opt_B3,y,train_label);


disp(['Accuracy of SVM for training set is ', num2str(round(accuracy_train,2)), '%']);
disp(['Accuracy of SVM for testing set is ', num2str(round(accuracy_test,2)), '%']);

function [label,accuracy] = get_accuracy_linear(data, opt_wt, opt_b,samples,Observedlabel)
counter = 0;
label = zeros(samples);
for i = 1:samples
    labels = opt_wt'*data(:,i) + opt_b;
    if(labels > 0)
        label(i) = 1;
    else
        label(i) = -1;
    end 
    if(label(i) == Observedlabel(i))
        counter = counter+1;
    end
end
 accuracy = (counter/samples)*100;
end

function [label,accuracy] = get_accuracy_poly(data, opt_wt, opt_b,samples,Observedlabel)
counter = 0;
[X,Y] = size(data);
x = sym('x',[1,X]);
label = zeros(samples);
for i = 1:samples
    labels = eval(subs(opt_wt,x,data(:,i)')) + opt_b;
    if(labels > 0)
        label(i) = 1;
    else
        label(i) = -1;
    end 
    if(label(i) == Observedlabel(i))
        counter = counter+1;
    end
end
 accuracy = (counter/samples)*100;
end
