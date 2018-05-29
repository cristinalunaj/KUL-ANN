%% PROBLEM 1- WHINE CLASSIFICATION
clc;
clear all;
close all;
%% CREATION OF SETS 
dataset = csvread('winequality-red_commaSeparated.csv',1);
%SELECTION OF THE CLASSES THAT WE ARE GONNA USE IN THE CLASSIFICATION TASK
[indexClassesRow,indexClassesCols]=find(dataset(:,end)==5 | dataset(:,end)==6);
binaryDS = dataset(indexClassesRow,:);

%RANDOMIZATION OF THE ROWS IN THE DS
numSamples = size(binaryDS,1);
load('permutation.mat');%randperm(numSamples);
binaryDS_random = binaryDS(permutation,:);

%SELECTION OF THE TEST SAMPLES, 98 OF C+ & 98 OF C-
test_set= [binaryDS_random(find(binaryDS_random(:,end)==5,98),:);binaryDS_random(find(binaryDS_random(:,end)==6,98),:)];
binaryDS_random(find(binaryDS_random(:,end)==5,98),:)=[];
binaryDS_random(find(binaryDS_random(:,end)==6,98),:)=[];


%CREATION OF VALIDATION AND TEST SETS
numSamples = size(binaryDS_random,1);
trainRatio = 0.83
valRatio = 0.17
testRatio = 0
[trainInd,valInd,testInd] = divideblock(numSamples,trainRatio,valRatio,testRatio);



train_set = binaryDS_random(trainInd,:);
validation_set = binaryDS_random(valInd,:);
avg_training = mean(train_set(:,1:end-1));
std_training = std(train_set(:,1:end-1));

% figure;
% histogram(train_set(:,end))
% hold on
% histogram(validation_set(:,end))
% hold on
% histogram(test_set(:,end))
% legend('train', 'val', 'test');

% %Split labels and features
% train_X = train_set(:,1:end-1)
% train_y= train_set(:,end)
% train_y_1hot = zeros(size(train_y,1),2);
% train_y_1hot(find(train_y(:,end)==5),1)=1;
% train_y_1hot(find(train_y(:,end)==6),2)=1;
% 
% 
% 
% 
% val_X = validation_set(:,1:end-1)
% val_y= validation_set(:,end)
% val_y_1hot = zeros(size(val_y,1),2);
% val_y_1hot(find(val_y(:,end)==5),1)=1;
% val_y_1hot(find(val_y(:,end)==6),2)=1;
% 
% X and y contain validation and trianing samples
X = binaryDS_random(:,1:end-1)
y = binaryDS_random(:,end)
y_1hot = zeros(size(y,1),2);
y_1hot(find(y(:,end)==5),1)=1;
y_1hot(find(y(:,end)==6),2)=1;

test_X = test_set(:,1:end-1)
test_y= test_set(:,end)
test_y_1hot = zeros(size(test_y,1),2);
test_y_1hot(find(test_y(:,end)==5),1)=1;
test_y_1hot(find(test_y(:,end)==6),2)=1;

%standarization: 
X = (X-avg_training)./std_training;
test_X = (test_X-avg_training)./std_training;

%save('sets.mat', 'X','y', 'test_X','test_y')


%% network
close all
neurons = [5,10,20,30,40,50]%[5,10,20,30,40,50,100,200]%,300,400]
trainingAlgorithms = ["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm","trainbr"]
%trainingAlgorithms = ["traingda","traincgf","trainbfg", "trainlm","trainbr"]
epochs = 2000
learning_rate = 0.00001

weights_layer1 = [];
weights_layer2 = []; 
weights_layer3 = []; 
bias1 = [];
bias2 = []; 
bias3 = []; 
results_matrix = zeros(7, 5);
training_results = struct('traingd',[],'traingda',[],'traincgf',[], 'traincgp',[],'trainbfg',[],'trainlm',[], 'trainbr',[]);

for attempt=1:10
    for numNeurons=neurons
        alg_index = 1;
        for alg=trainingAlgorithms
            close all
            %SAME INITIALIZATION
            numNeurons
            alg = char(alg)
            net = patternnet([numNeurons,floor(numNeurons/2)],alg);%alg
            net.divideFcn = 'divideblock';
            
            
           
            net.output.processFcns = {};
            net.performFcn = 'crossentropy';
            net.layers{1}.transferFcn = 'poslin';
            net.layers{2}.transferFcn = 'poslin';
            net.layers{3}.transferFcn = 'softmax';
            %net.performParam.regularization = 0.1;
            if(strcmp('traingd',alg))
                weights_layer1=net.iw{1,1};  %set the same weights and biases for the networks 
                weights_layer2=net.lw{2,1};
                weights_layer3=net.lw{3,1};
                bias1=net.b{1};
                bias2=net.b{2};
                bias3 = net.b{3};
            else
                net.iw{1,1}=weights_layer1;
                net.lw{2,1}=weights_layer2;
                net.lw{3,1}=weights_layer3;
                net.b{1} =bias1;
                net.b{2}=bias2;
                net.b{3}=bias3;
            end

            %training
            if(alg=="trainbr")
               net.trainParam.epochs=800;
               net.divideParam.trainRatio = 1;
%                net.divideParam.valRatio = 0;
%                net.divideParam.testRatio = 0;
               X_train = X(1:933,:);
               y_train1hot = y_1hot(1:933,:);
               net.trainParam.max_fail = 10;
            else
               net.trainParam.epochs=epochs;
               net.divideParam.trainRatio = trainRatio;
               net.divideParam.valRatio = valRatio;
               net.divideParam.testRatio = testRatio;
               X_train = X;
               y_train1hot=y_1hot;
            end
            
               
            %net.trainParam.epochs=epochs;
            %net.trainParam.lr = learning_rate;
            [net,tr] = train(net, X_train', y_train1hot');
            training_results = setfield(training_results,alg,tr);
            
            valIndex = tr.valInd;
            testIndex = tr.testInd;
            traininigIndex = tr.trainInd;
            
            if(isempty(valIndex)&&alg=="trainbr")
                predictions_val = (sim(net, X(934:1123,:)'));
                y_val1hot = y_1hot(934:1123,:);
            else
                predictions_val = (sim(net, X(valIndex,:)'));
                y_val1hot = y_1hot(valIndex,:);
            end
             
            %predictions: 
            predictions_test = (sim(net,test_X'));
            predictions_train = (sim(net, X(traininigIndex,:)'));
            
            
            [c,~,~,~]=confusion(predictions_test,test_y_1hot');
            ccr_test = 100*(1-c)
            %plotconfusion(test_y_1hot',predictions_test);
            [c_train,~,~,~]=confusion(predictions_train,y_1hot(traininigIndex,:)');
            ccr_train = 100*(1-c_train)
            %plotconfusion(y_1hot(traininigIndex,:)',predictions_train);
            [c_val,~,~,~]=confusion(predictions_val,y_val1hot');
            ccr_val = 100*(1-c_val)
            %plotconfusion(y_1hot(valIndex,:)',predictions_val);

            results_matrix(alg_index,:)=[ccr_train,ccr_val,ccr_test,tr.time(end),tr.epoch(end)];
            alg_index=alg_index+1; 
        end
        save(strcat('TEST3/relu/std_matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix');
        save(strcat('TEST3/relu/std_training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results');
    end
end


