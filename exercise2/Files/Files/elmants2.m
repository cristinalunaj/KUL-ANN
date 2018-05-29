% In this script an elman network is trained and tested in order to model a so called Hammerstein model. 
% The system is described like this:

% x(t+1) = 0.8x(t) + sin(u(t+1))
% y(t+1) = x(t+1);

% Elman network should be able to understand the relation between output
% y(t) and input u(t). x(t) is a latent variable representing the internal
% state of the system/

% Ricardo Castro-Garcia Feb-2017

%% Clean the workspace
clc;
clear;
close all;

%% Set the parameters of the run
n_training = [850,2125,4250,6375,8500];             % Number of training points (this includes training and validation).
n_te = 1500;             % Number of test points
neurons = [1,2,3,4,5,6,8,10,20,30];         % Number of neurons
n = 10000;               % Total number of samples
ne = 10000;               % Number of epochs
perc_training = 0.7;    % Number between 0 and 1. The validation set will be 1-perc_training.
n_repetitions = 10; 
% transferFunctions = [{'tansig', 'logsig'},{'tansig','tansig'},{'tansig','purelin'},...
%     {'logsig','tansig'},{'logsig','logsig'},{'logsig','purelin'}];
experiments_Matrix = [];

results_matrix = zeros(6, 6);

weights_layer1=[];
weights_layer2=[];
weights_layer1_internal = [];
bias1=[];
bias2=[];

index_parameter = 1;
for attempt = 1:n_repetitions
    for n_neurons = neurons
        for n_tr = n_training
            %n = n_tr+700;
%             for epochs=ne
        %for funct = transferFunctions
           
                if n < n_tr+n_te
                    n = n_tr+n_te;
                end

                if perc_training >= 1 || perc_training <= 0
                    error('The training set is ill defined. The variable perc_training should be between 0 and 1')
                end

                %% Create the samples
                % Allocate memory
                u = zeros(1, n);
                x = zeros(1, n);
                y = zeros(1, n);

                % Initialize u, x and y
                u(1)=randn; 
                x(1)=rand+sin(u(1));
                y(1)=x(1);

                % Calculate the samples
                for i=2:n
                    u(i)=randn;
                    x(i)=.8*x(i-1)+sin(u(i));
                    y(i)=x(i);
                end

                %% Create the datasets
                % Training set
                X=num2cell(u(1:n_tr)); 
                T=num2cell(y(1:n_tr));

                % Test set
                T_test=num2cell(y(end-n_te:end)); 
                X_test=num2cell(u(end-n_te:end));

                %% Train and simulate the network
                % Create the net and apply the selected parameters
                net = newelm(X,T,n_neurons);        % Create network
                %view(net)
                net.trainParam.epochs = ne;         % Number of epochs
                net.divideParam.testRatio = 0;
                net.divideParam.valRatio = 1-perc_training;
                net.divideParam.trainRatio = perc_training;
                
                if(index_parameter==1)
                    weights_layer1=net.iw{1,1};  %set the same weights and biases for the networks 
                    weights_layer1_internal=net.lw{1,1};
                    weights_layer2=net.lw{2,1};
                    bias1=net.b{1};
                    bias2=net.b{2};
                else
                    net.iw{1,1}=weights_layer1;
                    net.lw{1,1} = weights_layer1_internal;
                    net.lw{2,1}=weights_layer2;
                    net.b{1} =bias1;
                    net.b{2}=bias2
                end
                

                [net,tr] = train(net,X,T);               % Train network

                T_test_sim = sim(net,X_test);       % Test the network
                T_train_sim = sim(net,X)


                R_train = corrcoef(cell2mat(T),cell2mat(T_train_sim));
                R_train = R_train(1,2);
                my_MSE_train = mse(cell2mat(T)-cell2mat(T_train_sim));

                R = corrcoef(cell2mat(T_test),cell2mat(T_test_sim));
                R = R(1,2);
                my_MSE = mse(cell2mat(T_test)-cell2mat(T_test_sim));
                
                
                results_matrix(index_parameter,:)=[my_MSE_train,my_MSE,R_train,R,tr.time(end),tr.epoch(end)];
                index_parameter=index_parameter+1; 
                
                
            end
            save(strcat('matrix_results_',int2str(n_neurons),'_',int2str(attempt),'.mat'),'results_matrix')
            index_parameter=1;
%         end
     end
end

%save('Elman_experiments_Matrix.mat', 'experiments_Matrix');

%% Plot results
%Plot results and calculate correlation coefficient between target and
%output
% 
% figure;
% subplot 211
% n_test = size(X_test,2);
% plot(0:(n_test-1),cell2mat(T_test),'r',0:(n_test-1),cell2mat(T_test_sim),'b');
% xlabel('time');
% ylabel('y');
% legend('target','prediction','Location', 'southeast');

% R_train = corrcoef(cell2mat(T),cell2mat(T_train_sim));
% R_train = R_train(1,2);
% my_MSE_train = mse(cell2mat(T)-cell2mat(T_train_sim));
% 
% R = corrcoef(cell2mat(T_test),cell2mat(T_test_sim));
% R = R(1,2);
% my_MSE = mse(cell2mat(T_test)-cell2mat(T_test_sim));
% 


% title(['R = ' num2str(R) '. MSE = ' num2str(my_MSE)])
% xlim([0, n_test-1])
% subplot 212, 
% plot(cell2mat(T_test),cell2mat(T_test_sim),'or',cell2mat(T_test),cell2mat(T_test),'.b');
% title('Scatter plot')
% xlabel('Target')
% ylabel('Prediction')
% legend('Actual fit', 'Perfect fit', 'location', 'southeast')
% %--------------------------------------------------------------------------
