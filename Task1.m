%Task 1 - Train
clear; clc; 
load('train.mat'); %load training data

%pre-processing data 
std_train_data = standardisation(train_data);

[M,N] = size(std_train_data);

%%for finding H matrix
gram_matrix = zeros(N,N);
h_matrix = zeros(N,N);

C = 10^6; %really large value for hard margin
threshold = 10^-4; 

f = -ones(N,1);
A = [];
b = [];
Aeq = train_label';
beq = 0;
lb = zeros(N,1);
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);

%Uncomment the required kernel as required

%Hard margin using linear kernel
disp('Hard margin with Linear kernel');
disp('Calculating Gram Matrix');
[H_mat1, G_mat1] = linear_kernel(std_train_data,train_label,gram_matrix,h_matrix,N);
%check for mercer's condition
disp('Checking for mercer condition');
mercer(G_mat1,threshold);
%find alpha values 
disp('Starting Quadratic programming');
ub = ones(N,1) * C;
alpha1 = quad_progFunc(H_mat1,f,A,b,Aeq,beq,lb,ub,x0,options);
%Discriminant for Hard Margin With Linear Kernel
[opt_B1, opt_W1] = discriminant_linear(std_train_data, train_label,alpha1,threshold);
disp('Discriminant function calculated for Hard Margin with Linear Kernel');

%Hard Margin with Polynomial Kernel
% p = 5 ; 
% [H_mat2, G_mat2] = polynomial_kernel(std_train_data,train_label,gram_matrix,h_matrix,N,p);
% %check for mercer's condition
% disp('Checking for mercer condition');
% mercer(G_mat2,threshold);
% %find alpha values and optimal weights
% disp('Starting Quadratic programming');
% ub = ones(N,1) * C;
% alpha2 = quad_progFunc(H_mat2,f,A,b,Aeq,beq,lb,ub,x0,options);
% [opt_B2, opt_W2] = discriminant_poly_hard(std_train_data, train_label,alpha2,threshold,p); 
% disp('Discriminant function calculated for Hard Margin with Non-Linear(Polynomial) Kernel');
 
%Soft Margin with Polynomial Kernel
% p = 5; 
% C =2.1 ; % C = 0.6,1.1,2.1 try for all these different values. Uncomment as required
% [H_mat3, G_mat3] = polynomial_kernel(std_train_data,train_label,gram_matrix,h_matrix,N,p);
% %check for mercer's condition
% disp('Checking for mercer condition');
% mercer(G_mat3,threshold);
% %find alpha values and optimal weights
% disp('Starting Quadratic programming');
% ub = ones(N,1) * C;
% alpha3 = quad_progFunc(H_mat3,f,A,b,Aeq,beq,lb,ub,x0,options);
% [opt_B3, opt_W3] = discriminant_poly_soft(std_train_data, train_label,alpha3,threshold, p,C); 

% Save workspace variables for task 2
save('discriminant_variables');

%uncomment if running task 1 and task 2
run Task2.m

%calculating H and gram matrix from linear Kernel
function [h_mat,gram_mat] = linear_kernel(trainData,train_label,gram_mat,h_mat,noOfSamples)
    for i = 1:noOfSamples
        for j = 1:i
            gram_mat(i,j) = trainData(:,i)' * trainData(:,j);
            h_mat(i,j) = train_label(i) * train_label(j) * gram_mat(i,j);
            h_mat(j,i) = h_mat(i,j);    %because kernel is symmetric x1x2 = x2x1
            gram_mat(j,i) =  gram_mat(i,j);
        end
    end 
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
function alpha = quad_progFunc(H,f,A,b,Aeq,beq,lb,ub,x0,options)
disp('Calculating alpha values...');
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
end 

%calculating optimal weight and bias for hard margin with linear kernel
function [b_opt,W_opt] = discriminant_linear(train_data, train_label,alpha,threshold_alpha)
W_opt = 0; 
disp('Finding optimal weights');
for i = 1:size(alpha)
    if(alpha(i) < threshold_alpha)
        alpha(i)=0; %rounding very small values of alpha to zero
    end  
    W_opt = W_opt + (alpha(i) * train_label(i) * train_data(:,i));
end 
disp('Calculating optimal bias for hard margin'); 
SV = find(alpha > 0); %SV contains all the indices where alpha is not zero. 
b_opt = 1/train_label(SV(1))- (W_opt' * train_data(:,SV(1)));
end 

%calculating optimal weight and bias for hard margin with polynomial kernel
function [B_opt,W_opt] = discriminant_poly_hard(train_data, train_label,alpha,threshold_alpha,p)
[X,Y] = size(train_data);
W = sym('W',[1,Y]);
W_opt = sym('W_opt');
x = sym('x',[1,X]);
disp('Finding optimal weights');
for i = 1:size(alpha) 
    if(alpha(i) < threshold_alpha)
        alpha(i)=0; %rounding very small values of alpha to zero
    end 
    temp1 = alpha(i) * train_label(i);
    W(1,i) = temp1 .*(1+(x*train_data(:,i)))^p;
end
W_opt = sum(W,'all');
disp('Calculating optimal bias for hard margin');
SV = find(alpha > 0); %SV contains all the indices where alpha is not zero.
b_opt = zeros(size(SV));
disp(size(SV));
for i = 1:size(SV)
 disp(i);
 b_opt(i) = 1/train_label(SV(i))- eval(subs(W_opt, x,train_data(:,SV(i))'));
 end 
B_opt = mean(b_opt);
end 

%calculating optimal bias for soft margin
function [B_opt, W_opt] = discriminant_poly_soft(train_data, train_label,alpha,threshold_alpha,p,C)
[X,Y] = size(train_data);
W = sym('W',[1,Y]);
W_opt = sym('W_opt');
x = sym('x',[1,X]);
disp('Finding optimal weights');
for i = 1:size(alpha) 
    if(alpha(i) < threshold_alpha)
        alpha(i)=0; %rounding very small values of alpha to zero
    end 
    temp1 = alpha(i) * train_label(i);
    W(1,i) = temp1 .*(1+(x*train_data(:,i)))^p;
 end
W_opt = sum(W,'all');

disp('Calculating optimal bias for soft margin');
SV = find(alpha > 0 & alpha < C); %SV contains all the indices where alpha is not zero. 
b_opt = zeros(size(SV));
for i = 1:size(SV)
 b_opt(i) = 1/train_label(SV(i))- eval(subs(W_opt, x,train_data(:,SV(i))'));
 end 
B_opt = mean(b_opt);
end 

