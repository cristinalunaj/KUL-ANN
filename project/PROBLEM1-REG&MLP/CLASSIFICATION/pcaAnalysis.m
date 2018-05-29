%% PCA
clc; clear; close all

load("sets.mat");
X_train = X(1:933,:);
X_val = X(934:1123,:);
y_train = y(1:933,:);
avg = mean(X_train,1)
std11 = std(X_train,1)

X_trainingNew = (X_train-avg)./std11;
X_valNew = (X_val-avg)./std11;
X_testNew = (test_X-avg)./std11;


covMatrix = cov(X_trainingNew);
[v1,d1] = eig(covMatrix);
eigenValues = diag(d1)';
newVector = zeros(size((eigenValues)));
for i=1:11
    newVector(end-i+1) = eigenValues(i)
end
cummulativeEigneValue= cumsum(newVector); 
totalSum = sum(newVector)
b1 = bar((newVector/totalSum)*100);
b1.FaceAlpha = 1;hold on;
%b2 = bar((cummulativeEigneValue/(cummulativeEigneValue(end)))*100)
%b2.FaceAlpha = 0.4;hold on;
title('Eigenvalues')
xlabel('Eigenvalues')
ylabel('Amount of variance information(%)')
%legend({'Variance percentage per eigenvalue','Cummulative variance'},'Location','northwest')
% saveas(gcf,strcat('PCAplots/eigenvalues.jpg'))
% savefig(strcat('PCAplots/eigenvalues.fig'))


errorVector = [];

for q=8
    %zeroMeanX = (X_train-avg)';
    %Training data
    [v,d] = eigs(covMatrix,(q));
    E = v;
    
    z_train = (E'*X_trainingNew');
    z_val = (E'*X_valNew');
    z_test = (E'*X_testNew');
    
    %reconstruction
    z_train_hat = (E*z_train);
    error = mse(z_train_hat-X_trainingNew');
    errorVector=[errorVector;error];
    
end
% figure;
%newError = (errorVector/sum(errorVector))*100;
% plot(errorVector, 'b+-');
% title('Reconstruction error - MSE')
% xlabel('Number of components')
% ylabel('MSE')
% saveas(gcf,strcat('PCAplots/reconstructionError.jpg'))
% savefig(strcat('PCAplots/reconstructionError.fig'))
close all

X = [z_train';z_val'];
y_1hot = zeros(size(y,1),2);
y_1hot(find(y(:,end)==5),1)=1;
y_1hot(find(y(:,end)==6),2)=1;

test_X = z_test';
test_y_1hot = zeros(size(test_y,1),2);
test_y_1hot(find(test_y(:,end)==5),1)=1;
test_y_1hot(find(test_y(:,end)==6),2)=1;

% save('setsPCA_k8.mat', 'X','y_1hot', 'test_X','test_y_1hot')

%     xhat = (E*z)';

    %xhat = (xhat.*std11)+avg;
    
%     error = (sqrt(mean(mean((inputX-xhat).^2))));
%     errorVector = [errorVector;error];

% end
%figure
%bar(diag(d1));

%% network 1layer
% close all
% trainRatio = 0.83
% valRatio = 0.17
% testRatio = 0
% neurons = [10,20,30,40,50]%[5,10,20,30,40,50,100,200]%,300,400]
% trainingAlgorithms = ["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm","trainbr"]
% %trainingAlgorithms = ["traingda","traincgf","trainbfg", "trainlm","trainbr"]
% epochs = 2000
% learning_rate = 0.00001
% 
% weights_layer1 = [];
% weights_layer2 = []; 
% weights_layer3 = []; 
% bias1 = [];
% bias2 = []; 
% bias3 = []; 
% results_matrix = zeros(7, 5);
% training_results = struct('traingd',[],'traingda',[],'traincgf',[], 'traincgp',[],'trainbfg',[],'trainlm',[], 'trainbr',[]);
% 
% for attempt=1:10
%     for numNeurons=neurons
%         alg_index = 1;
%         for alg=trainingAlgorithms
%             close all
%             %SAME INITIALIZATION
%             numNeurons
%             alg = char(alg)
%             net = patternnet(numNeurons,alg);%alg
%             net.divideFcn = 'divideblock';
%             %net.trainParam.max_fail = 5;
%             net.layers{2}.transferFcn = 'softmax';
%            
%             net.output.processFcns = {};
%             net.performFcn = 'crossentropy';
%             net.layers{1}.transferFcn = 'poslin';
%             %net.performParam.regularization = 0.1;
%             if(strcmp('traingd',alg))
%                 weights_layer1=net.iw{1,1};  %set the same weights and biases for the networks 
%                 weights_layer2=net.lw{2,1};
%                 weights_layer3=net.lw{3,1};
%                 bias1=net.b{1};
%                 bias2=net.b{2};
%                  bias3 = net.b{3};
%             else
%                 net.iw{1,1}=weights_layer1;
%                 net.lw{2,1}=weights_layer2;
%                  net.lw{3,1}=weights_layer3;
%                 net.b{1} =bias1;
%                 net.b{2}=bias2;
%                  net.b{3}=bias3;
%             end
% 
%             %training
%             if(alg=="trainbr")
%                net.trainParam.epochs=800;
%                net.divideParam.trainRatio = 1;
% %                net.divideParam.valRatio = 0;
% %                net.divideParam.testRatio = 0;
%                X_train = X(1:933,:);
%                y_train1hot = y_1hot(1:933,:);
%                net.trainParam.max_fail = 10;
%             else
%                net.trainParam.epochs=epochs;
%                net.divideParam.trainRatio = trainRatio;
%                net.divideParam.valRatio = valRatio;
%                net.divideParam.testRatio = testRatio;
%                X_train = X;
%                y_train1hot=y_1hot;
%             end
%             
%                
%             %net.trainParam.epochs=epochs;
%             %net.trainParam.lr = learning_rate;
%             [net,tr] = train(net, X_train', y_train1hot');
%             training_results = setfield(training_results,alg,tr);
%             
%             valIndex = tr.valInd;
%             testIndex = tr.testInd;
%             traininigIndex = tr.trainInd;
%             
%             if(isempty(valIndex)&&alg=="trainbr")
%                 predictions_val = (sim(net, X(934:1123,:)'));
%                 y_val1hot = y_1hot(934:1123,:);
%             else
%                 predictions_val = (sim(net, X(valIndex,:)'));
%                 y_val1hot = y_1hot(valIndex,:);
%             end
%              
%             %predictions: 
%             predictions_test = (sim(net,test_X'));
%             predictions_train = (sim(net, X(traininigIndex,:)'));
%             
%             
%             [c,~,~,~]=confusion(predictions_test,test_y_1hot');
%             ccr_test = 100*(1-c)
%             %plotconfusion(test_y_1hot',predictions_test);
%             [c_train,~,~,~]=confusion(predictions_train,y_1hot(traininigIndex,:)');
%             ccr_train = 100*(1-c_train)
%             %plotconfusion(y_1hot(traininigIndex,:)',predictions_train);
%             [c_val,~,~,~]=confusion(predictions_val,y_val1hot');
%             ccr_val = 100*(1-c_val)
%             %plotconfusion(y_1hot(valIndex,:)',predictions_val);
% 
%             results_matrix(alg_index,:)=[ccr_train,ccr_val,ccr_test,tr.time(end),tr.epoch(end)];
%             alg_index=alg_index+1; 
%         end
%         save(strcat('PCA_TEST/k',num2str(q),'/relu/1layer/matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix');
%         save(strcat('PCA_TEST/k',num2str(q),'/relu/1layer/training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results');
%     end
% end




%% train nw - 2 layer
close all
trainRatio = 0.83
valRatio = 0.17
testRatio = 0
neurons = [10,20,30,40,50]%[5,10,20,30,40,50,100,200]%,300,400]
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
            %net.trainParam.max_fail = 5;
            net.layers{1}.transferFcn = 'poslin';
            net.layers{2}.transferFcn = 'poslin';
            net.layers{3}.transferFcn = 'softmax';
           
            net.output.processFcns = {};
            net.performFcn = 'crossentropy';
            
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
        save(strcat('PCA_TEST/k',num2str(q),'/relu/2layer/matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix');
        save(strcat('PCA_TEST/k',num2str(q),'/relu/2layer/training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results');
    end
end


%% %% train nw - 3 layer
close all
trainRatio = 0.83
valRatio = 0.17
testRatio = 0
neurons = [10,20,30,40,50]%[5,10,20,30,40,50,100,200]%,300,400]
trainingAlgorithms = ["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm","trainbr"]
%trainingAlgorithms = ["traingda","traincgf","trainbfg", "trainlm","trainbr"]
epochs = 2000
learning_rate = 0.00001

weights_layer1 = [];
weights_layer2 = []; 
weights_layer3 = []; 
weights_layer4 = [];
bias1 = [];
bias2 = []; 
bias3 = []; 
bias4 = []; 
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
            net = patternnet([numNeurons,floor(numNeurons/2),floor(numNeurons/4)],alg);%alg
            net.divideFcn = 'divideblock';
            %net.trainParam.max_fail = 5;
            net.layers{1}.transferFcn = 'poslin';
            net.layers{2}.transferFcn = 'poslin';
            net.layers{3}.transferFcn = 'poslin';
            net.layers{4}.transferFcn = 'softmax';
           
            net.output.processFcns = {};
            net.performFcn = 'crossentropy';
            
            %net.performParam.regularization = 0.1;
            if(strcmp('traingd',alg))
                weights_layer1=net.iw{1,1};  %set the same weights and biases for the networks 
                weights_layer2=net.lw{2,1};
                weights_layer3=net.lw{3,1};
                weights_layer4=net.lw{4,1};
                bias1=net.b{1};
                bias2=net.b{2};
                 bias3 = net.b{3};
                 bias4 = net.b{4};
            else
                net.iw{1,1}=weights_layer1;
                net.lw{2,1}=weights_layer2;
                 net.lw{3,1}=weights_layer3;
                 net.lw{4,1}=weights_layer4;
                net.b{1} =bias1;
                net.b{2}=bias2;
                 net.b{3}=bias3;
                 net.b{4}=bias4;
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
        save(strcat('PCA_TEST/k',num2str(q),'/relu/3layer/matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix');
        save(strcat('PCA_TEST/k',num2str(q),'/relu/3layer/training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results');
    end
end
