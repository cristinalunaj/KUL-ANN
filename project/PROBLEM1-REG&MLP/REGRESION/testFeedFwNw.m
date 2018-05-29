clc,clear,close all
%numNeurons = [20,30,40,50,100,200]; 
neurons = [20,30,40,50,100,200]
trainingAlgorithms = ["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm","trainbr"]
epochs = 2000
x_train = 0:0.05:3*pi;
y_train = sin(x_train.^2);%+0.3*randn(1,size(x_train,2));
% 
% x_val = 0:0.2:3*pi;
% y_val = sin(x_val.^2)

x_test = 0:0.02:3*pi;
y_test = sin(x_test.^2);

P_train=con2seq(x_train);
T_train = con2seq(y_train);

% P_val = con2seq(x_val);%inputs for predictions
% T_val = con2seq(y_val);%Test target

P_test = con2seq(x_test);%inputs for predictions
T_test = con2seq(y_test);%Test target

weights_layer1 = [];
weights_layer2 = []; 
bias1 = [];
bias2 = []; 
results_matrix = zeros(7, 6);
training_results = struct('traingd',[],'traingda',[],'traincgf',[], 'traincgp',[],'trainbfg',[],'trainlm',[], 'trainbr',[]);

for attempt=1:10
    for numNeurons=neurons
        alg_index = 1;
        for alg=trainingAlgorithms
            close all
            %SAME INITIALIZATION
            numNeurons
            alg = char(alg)
            net = feedforwardnet(numNeurons,alg);
            net.divideFcn='dividerand'
            %net.layers{1}.transferFcn = 'logsig'
            if(strcmp('traingd',alg))
                weights_layer1=net.iw{1,1};  %set the same weights and biases for the networks 
                weights_layer2=net.lw{2,1};
                bias1=net.b{1};
                bias2=net.b{2};
            else
                net.iw{1,1}=weights_layer1;
                net.lw{2,1}=weights_layer2;
                net.b{1} =bias1;
                net.b{2}=bias2
            end

            %training
            net.trainParam.epochs=epochs;
            [net,tr] = train(net, P_train, T_train);
            training_results = setfield(training_results,alg,tr)
            
            %predictions: 
            predictions_test = (sim(net,P_test));
            predictions_train = (sim(net, P_train));
            mse_train = mse_calc(cell2mat(predictions_train), cell2mat(T_train))
            mse_test = mse_calc(cell2mat(predictions_test), cell2mat(T_test))

            % figure
            % subplot(2,2,1)
            % plot(x_train,y_train,'bx',x_train,cell2mat(predictions_train),'r');
            % title('TRAINING - LABELSvsPRED')
            % subplot(2,2,2)
            [m,b,r_train]=postregm(cell2mat(predictions_train),y_train);
            % title('TRAINING - R')
            % subplot(2,2,3)
            % plot(x_test,y_test,'bx',x_test,cell2mat(predictions_test),'r');
            % title('TEST - LABELSvsPRED')
            % subplot(2,2,4)
            [m,b,r_test]=postregm(cell2mat(predictions_test),y_test);
            % title('TESTING - R')re
            results_matrix(alg_index,:)=[mse_train,mse_test,r_train,r_test,tr.time(end),tr.epoch(end)];
            alg_index=alg_index+1; 
        end
        save(strcat('matrix_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'results_matrix')
        save(strcat('training_results_',int2str(numNeurons),'_',int2str(attempt),'.mat'),'training_results')
    end
end



