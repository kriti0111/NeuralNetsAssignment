%pre-processing data - returns the standardised data

function transformed_data = standardisation(data)
mean_features = mean(data,2); %calculate the mean of each feature from 2000 samples
disp(mean_features);
stdDev_features = std(data,0,2); %calculate the standard deviation along each row
[M,N] = size(data);
transformed_data = zeros(M,N);
for i = 1:M
    for j = 1:N
        transformed_data(i,j) = (data(i,j)-mean_features(i))/stdDev_features(i);
    end
end
