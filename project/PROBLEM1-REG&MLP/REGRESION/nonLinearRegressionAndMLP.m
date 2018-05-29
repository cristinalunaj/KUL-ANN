%% DEFINITION OF DS:  student number: r0693025
clc;
clear;
close all
load('Data_Problem1_regression.mat')
d1 = 9;
d2 = 6;
d3 = 5;
d4 = 3;
d5 = 2;
Tnew = (d1*T1+d2*T2+d3*T3+d4*T4+d5*T5)/(d1+d2+d3+d4+d5);
inputVector = [X1; X2];
xSpace = linspace(0,1,13600)
totalNumberSamples = 13600;
%TRAINING

load("permutation.mat");
load("index.mat");
%x = randperm(13600,3000);
% [trainInd,valInd,testInd] = dividerand(3000,1/3,1/3,1/3)

X1_downSampl = X1(x);
X2_downSampl = X2(x);
xSpace_downSampl = xSpace(x);
T_downSampl = Tnew(x);


X1_trainig = X1_downSampl(trainInd);
X2_training = X2_downSampl(trainInd);
x_tr = xSpace_downSampl(trainInd);
T_training = T_downSampl(trainInd);


X1_val =  X1_downSampl(valInd);
X2_val = X2_downSampl(valInd);
x_val = xSpace_downSampl(valInd);
T_val = T_downSampl(valInd);


X1_test = X1_downSampl(testInd);
X2_test = X2_downSampl(testInd);
x_test = xSpace_downSampl(testInd);
T_test = T_downSampl(testInd);


% % FIGURE OF THE DATSET DISTRIBUTION IN TERMS OF X AXIS
% figure
% histogram(x_tr, 20)
% hold on
% histogram(x_val, 20)
% hold on
% histogram(x_test, 20)
% hold on
% legend('train','val','test')
% title('Distribution of sets in temporal axis')

%TRAIN
% f = scatteredInterpolant(X1_trainig,X2_training,T_training)
x1lim_train = linspace(0,1,1000);
x2lim_train = linspace(0,1,1000);
% [X_train,Y_train] = meshgrid(x1lim_train, x2lim_train);
% Z_training = f(X_train,Y_train);
% figure
% mesh(X_train,Y_train,Z_training);
% title('train')

%VALIDATION
% f = scatteredInterpolant(X1_val,X2_val,T_val)
% [X_val,Y_val] = meshgrid(x1lim_train, x2lim_train);
% Z_val = f(X_val,Y_val);
% figure
% mesh(X_val,Y_val,Z_val);
% title('validation')

%TEST
% f = scatteredInterpolant(X1_test,X2_test,T_test)
% [X_test,Y_test] = meshgrid(x1lim_train, x2lim_train);
% Z_test = f(X_test,Y_test);
% figure
% mesh(X_test,Y_test,Z_test);
% title('test')


%% CREATION OF NW : https://nl.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html
% info about algorithms: https://nl.mathworks.com/help/nnet/ug/train-and-apply-multilayer-neural-networks.html
X = [X1_trainig,X2_training;
     X1_val,X2_val;
     X1_test,X2_test];
    
Y = [T_training;
    T_val; 
    T_test];

close all
neurons = [50]%,40,50,100,200]%,300,400]
trainingAlgorithms = ["trainbr"]%["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm"]
epochs = 200

weights_layer1 = [];
weights_layer2 = []; 
bias1 = [];
bias2 = []; 
results_matrix = zeros(6, 7);
training_results =struct('trainbr',[]); %struct('traingd',[],'traingda',[],'traincgf',[], 'traincgp',[],'trainbfg',[],'trainlm',[]);
X_input = X';%(X')
Y_input = Y';%con2seq(Y')
hiddenFunction = 'tansig'

for attempt=1:1
    for numNeurons=neurons
        alg_index = 1;
        for alg=trainingAlgorithms
            %SAME INITIALIZATION
            numNeurons
            alg = char(alg)
            net = feedforwardnet(numNeurons,alg);
            net.trainParam.max_fail=3;
            %net.trainParam.min_grad=5e-5;
            net.trainParam.epochs = epochs;
            net.layers{1}.transferFcn = hiddenFunction;
            %net.trainParam.mu_max=1000
            %net.trainParam.min_grad = 1e-6
%             net.performParam.regularization = 0.00001
            net.divideFcn = 'divideblock';
            net.divideParam.trainRatio = 1/3;
            net.divideParam.valRatio = 1/3;
            net.divideParam.testRatio = 1/3;
            %net.layers{1}.transferFcn = 'logsig'
%             if(strcmp('traingd',alg))
%                 weights_layer1=net.iw{1,1};  %set the same weights and biases for the networks 
%                 weights_layer2=net.lw{2,1};
%                 bias1=net.b{1};
%                 bias2=net.b{2};
%             else
%                 net.iw{1,1}=weights_layer1;
%                 net.lw{2,1}=weights_layer2;
%                 net.b{1} =bias1;
%                 net.b{2}=bias2
%             end

            %training
            [net,tr] = train(net, X_input, Y_input);
            %view(net)
            valIndex = tr.valInd;
            testIndex = tr.testInd;
            traininigIndex = tr.trainInd;
            
            
            training_results = setfield(training_results,alg,tr)
            
            %predictions: 
            
            predictions_train = (sim(net, X_input(:,traininigIndex)));
            predictions_val = (sim(net,X_input(:,valIndex)));
            predictions_test = (sim(net,X_input(:,testIndex)));
%             plot(predictions_test, 'b');
%             hold on
%             plot(Y_input(:,testIndex), 'r');
%             legend('train', 'test');
            
            %Measures
            mse_train = mse_calc((predictions_train),(Y_input(:,traininigIndex)))
            mse_val = mse_calc((predictions_val),(Y_input(:,valIndex)))
            mse_test = mse_calc((predictions_test), (Y_input(:,testIndex)))
            
            [m,b,r_train]=postregm((predictions_train),Y(traininigIndex,:)');
            [m,b,r_test]=postregm((predictions_test),Y(testIndex,:)');
            
            results_matrix(alg_index,:)=[mse_train,mse_val,mse_test,r_train,r_test,tr.time(end),tr.epoch(end)];
            alg_index=alg_index+1; 
            close all
            
        end
          save(strcat('finalNw/matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix')
          save(strcat('finalNw/training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results')
    
%         save(strcat('RESULTS_nw/trainbr/matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix')
%         save(strcat('RESULTS_nw/trainbr/training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results')
    end
end

%% paint test predictions


f = scatteredInterpolant(X1_test,X2_test,predictions_test')
[X_test,Y_test] = meshgrid(x1lim_train, x2lim_train);
Z_test = f(X_test,Y_test);
figure
mesh(X_test,Y_test,Z_test);
title('test predictions')


% for early stopping: 
%net.trainParam.max_fail = 5;
% 
% %training
% net = train(net, X, Y);
% 
% %predictions: 
% a = sim(net, P_test);
% Y=net(P_test);

%[m,b,r]=postreg(a,T_test)
