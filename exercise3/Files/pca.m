%% DATA GENERATION: 
clc; clear, close all
%x = randn(50,500); %500 xpoints of dimension 50
x = load('choles_all');
x = x.p;
imagesc(x);
%imagesc(x)

kVectors = [1,2,5,10,15,20]; 
errorVector = [];
for lambda=1:6
    avg = mean(x,2);
    zeroMeanX = x-avg;
    covMatrix = cov(x');
    [v1,d1] = eig(covMatrix);
    bar(diag(d1));
    [v,d] = eigs(covMatrix,kVectors(lambda));
    E = v;

    xhat = (E'*zeroMeanX);
    xhat = E*xhat;

    xhat = xhat+avg;

    error = (sqrt(mean(mean((x-xhat).^2))));
    errorVector = [errorVector;error];
    %imagesc(xhat)
end
figure
bar(diag(d1));
figure
plot(kVectors, errorVector);


%% mapstd and processpca test
clc; clear, close all
%x = randn(50,500); %500 xpoints of dimension 50
x = load('choles_all');
x = x.p;
imagesc(x);
%imagesc(x)
[x,settings] = mapstd(x)
[z,PS]= processpca(x, 0.001)
xhat = processpca('reverse',z,PS)

%xhat = mapstd('reverse', xhat, settings);
error = (sqrt(mean(mean((x-xhat).^2))));
