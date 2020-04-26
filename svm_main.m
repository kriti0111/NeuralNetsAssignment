%% Task 3 - Evaluate 

load ('eval.mat');
load('train.mat'); %load training data


%pre-processing data 
strd_train_data = standardisation(train_data);
strd_eval_data = standardisation(eval_data);

[M,N] = size(strd_train_data);
gram_matrix = zeros(N,N);
h_matrix = zeros(N,N);
p = 1; 
C =0.1 ; 
threshold = 10^-4; 

[H_mat, G_mat] = polynomial_kernel(strd_train_data,train_label,gram_matrix,h_matrix,N,p);
%check for mercer's condition
disp('Checking for mercer condition');
mercer(G_mat,threshold);
%find alpha values and optimal weights
disp('Starting Quadratic programming');
ub = ones(N,1) * C;
alpha = quad_progFunc(H_mat,f,A,b,Aeq,beq,lb,ub,x0,options, threshold);
opt_B = discriminant_func(strd_train_data, train_label,alpha, p,C); 

 eval_predicted = getlabel(strd_train_data,strd_eval_data,alpha, train_label, p,opt_B);
 disp('Predicted labels');
 disp(eval_predicted);
 
 %uncomment if eval_label is avaialable
 %eval_acc = getaccuracy(eval_label, eval_predicted);

%disp(['Accuracy of SVM for evaluation set is ', num2str(round(eval_acc,2)), '%']);

%calculating optimal bias for soft margin
function bias = discriminant_func(train_data, train_label,alpha,p,C)
disp('Calculating optimal bias for soft margin');
SV = find(alpha > 0 & alpha < C); %SV contains all the indices where alpha is not zero. 
b_opt = zeros(size(SV));
for j = 1:size(SV)
s_idx = SV(j);
weights = 0;
 for i = 1:size(train_data,2)
 weights = weights + alpha(i,:) * train_label(i,:) * (train_data(:,s_idx)' * train_data(:,i) + 1) ^ p;
 end
 b_opt(j) = train_label(s_idx,:) - weights;
end
bias = mean(b_opt);
end 

function prediction = getlabel(train_data,data,alpha, train_label,p,bias)
samples = size(data,2);
N = size(train_data,2);
prediction = zeros(samples,1);
gx = zeros(samples);
for j = 1:samples        
    weight = 0;
    for i = 1:N     
        weight = weight + alpha(i,:) * train_label(i,:) * (data(:,j)' * train_data(:,i) + 1) ^ p;
    end
    gx(j) = weight + bias;
end
        
for i = 1:samples
    if gx(i) > 0
       prediction(i,1) = 1;  
    else
       prediction(i,1) = -1; 
    end
end

end

function accuracy = getaccuracy(label, predicted_label)
counter = 0; 
samples = size(predicted_label,1);
for i = 1:size(predicted_label,1)
 if(label(i) == predicted_label(i))
        counter = counter+1;    
 end 
end
accuracy = (counter/samples)*100;
end

%calculating H and gram matrix from polynomial Kernel
function [h_mat,gram_mat] = polynomial_kernel(trainData,train_label,gram_mat,h_mat,noOfSamples,p)
disp(['Polynomial kernel , p = ',num2str(p)]);
disp('Calculating Gram Matrix');
    for i = 1:noOfSamples
        for j = 1:i
            gram_mat(i,j) = ((trainData(:,i)' * trainData(:,j)) + 1)^p;
            h_mat(i,j) = train_label(i) * train_label(j) * gram_mat(i,j);
            h_mat(j,i) = h_mat(i,j);    %because kernel is symmetric x1x2 = x2x1
            gram_mat(j,i) =  gram_mat(i,j);
        end
    end 
end 

%checking for mercer condition
function eigen_values = mercer(gramMatrix,threshold)
eigen_values = eig(gramMatrix);
I = find(eigen_values < 0);
if(isempty(I))
    disp('mercer condition passed');
else
    Abs_I = find(abs(eigen_values(I)) > threshold);
    if(isempty(Abs_I))
    disp('mercer condition passed');
    else
        disp('mercer condition failed');
    end 
end
end 

%calculating alpha values
function alpha = quad_progFunc(H,f,A,b,Aeq,beq,lb,ub,x0,options,threshold_alpha)
disp('Calculating alpha values...');
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
for i = 1:size(alpha) 
    if(alpha(i) < threshold_alpha)
        alpha(i)=0; %rounding very small values of alpha to zero
    end 
end
end 