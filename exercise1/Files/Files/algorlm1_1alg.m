clear
clc
close all
nnet.guis.closeAllViews()

%%%%%%%%%%%
%algorlm.m -> TRAINING 1 ALGORITHM
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%
%DATA%%%%%%%%
%generation of examples and targets
x=0:0.05:3*pi; y=sin(x.^2)+0.3*randn(1,size(x,2));

%x_test = 3*pi:0.05:2*pi; y_test=sin(x_test.^2);% AÑADIR RUIDO????

algorithm1 = 'traingd';
algorithm2 = 'trainbfg';
algorithm3 = 'trainlm';
algorithm4 = 'traincgf';
algorithm5 = 'traincgp';
algorithm6 = 'traingda';


neurons = 100;
epochs_start = 15;
epochs_middle = 50;
epochs_end = 500;
%attempt = 3;

results_matrix = zeros(6, 8);
training_results = struct('net1',[],'net2',[],'net3',[], 'net4',[],'net5',[],'net6',[]);
%noisy data: 
%x=0:0.05:3*pi; y=sin(x.^2)+0.01*randn(1,189);
%%%%%%%%%%%%%%%%%
p=con2seq(x); t=con2seq(y); % convert the data to a useful format

for attempt=1:1
    %INITIALIZATION WEIGHTS
    randow_weigths_layer1 = rand(neurons,1)
    randow_weigths_layer2 = rand(1,neurons)
    random_bias_layer1 = rands(neurons);


    %CREATION OF NWS & INITIALIZATION
    net1=feedforwardnet(neurons,algorithm1);
    net1=init(net1)
    net1 = configure(net1,p,t);
    % net1.IW{1,1} = randow_weigths_layer1;
    % net1.b{1,1} = random_bias_layer1;
    % net1.LW{2,1} = randow_weigths_layer2;
    % net1.b{2,1} = 0;
    %NET2
    net2=feedforwardnet(neurons,algorithm2);
    net2=init(net2)
    net2 = configure(net2,p,t);
    net2.IW{1,1} = net1.IW{1,1};
    net2.b{1,1} = net1.b{1,1}
    net2.LW{2,1} = net1.LW{2,1}
    net2.b{2,1} = net1.b{2,1};
    %NET3
    net3=feedforwardnet(neurons,algorithm3);
    net3=init(net3)
    net3 = configure(net3,p,t);
    net3.IW{1,1} = net1.IW{1,1};
    net3.b{1,1} = net1.b{1,1}
    net3.LW{2,1} = net1.LW{2,1}
    net3.b{2,1} = net1.b{2,1};
    %NET4
    net4=feedforwardnet(neurons,algorithm4);
    net4=init(net4)
    net4 = configure(net4,p,t);
    net4.IW{1,1} = net1.IW{1,1};
    net4.b{1,1} = net1.b{1,1}
    net4.LW{2,1} = net1.LW{2,1}
    net4.b{2,1} = net1.b{2,1};
    %NET5
    net5=feedforwardnet(neurons,algorithm5);
    net5=init(net5)
    net5 = configure(net5,p,t);
    net5.IW{1,1} = net1.IW{1,1};
    net5.b{1,1} = net1.b{1,1}
    net5.LW{2,1} = net1.LW{2,1}
    net5.b{2,1} = net1.b{2,1};
    %NET6
    net6=feedforwardnet(neurons,algorithm6);
    net6=init(net6)
    net6 = configure(net6,p,t);
    net6.IW{1,1} = net1.IW{1,1};
    net6.b{1,1} = net1.b{1,1}
    net6.LW{2,1} = net1.LW{2,1}
    net6.b{2,1} = net1.b{2,1};
    %END NW INITIALIZATION
    %wb = getwb(net1) %FOR OBTAINING WEIGHTS AND BIASES OF NW


    %TRAINING & SIMULATIONS
    net1.trainParam.epochs=epochs_start;  % set the number of epochs for the training 
    [net1,tr]=train(net1,p,t);   % train the networks with batch
    a11=sim(net1,p); %a21=sim(net2,p);  % simulate the networks with the input vector p

    net1.trainParam.epochs=epochs_middle;
    [net1,tr]=train(net1,p,t);
    a12=sim(net1,p);

    net1.trainParam.epochs=epochs_end;
    [net1,tr]=train(net1,p,t);
    a13=sim(net1,p);
    results_matrix(1,4:end)=[tr.time(end);tr.gradient(1);tr.gradient(end);tr.perf(1);tr.perf(end)]
    training_results.net1 = tr;
    %NET2
    net2.trainParam.epochs=epochs_start;  % set the number of epochs for the training 
    [net2,tr]=train(net2,p,t);   % train the networks with batch
    a21=sim(net2,p); %a21=sim(net2,p);  % simulate the networks with the input vector p

    net2.trainParam.epochs=epochs_middle;
    [net2,tr]=train(net2,p,t);
    a22=sim(net2,p);

    net2.trainParam.epochs=epochs_end;
    [net2,tr]=train(net2,p,t);
    a23=sim(net2,p);
    results_matrix(2,4:end)=[tr.time(end);tr.gradient(1);tr.gradient(end);tr.perf(1);tr.perf(end)]
    training_results.net2 = tr;
    %NET3
    net3.trainParam.epochs=epochs_start;  % set the number of epochs for the training 
    [net3,tr]=train(net3,p,t);   % train the networks with batch
    a31=sim(net3,p); %a21=sim(net2,p);  % simulate the networks with the input vector p

    net3.trainParam.epochs=epochs_middle;
    [net3,tr]=train(net3,p,t);
    a32=sim(net3,p);

    net3.trainParam.epochs=epochs_end;
    [net3,tr]=train(net3,p,t);
    a33=sim(net3,p);
    results_matrix(3,4:end)=[tr.time(end);tr.gradient(1);tr.gradient(end);tr.perf(1);tr.perf(end)]
    training_results.net3 = tr;
    %NET4
    net4.trainParam.epochs=epochs_start;  % set the number of epochs for the training 
    [net4,tr]=train(net4,p,t);   % train the networks with batch
    a41=sim(net4,p); %a21=sim(net2,p);  % simulate the networks with the input vector p

    net4.trainParam.epochs=epochs_middle;
    [net4,tr]=train(net4,p,t);
    a42=sim(net4,p);

    net4.trainParam.epochs=epochs_end;
    [net4,tr]=train(net4,p,t);
    a43=sim(net4,p);
    results_matrix(4,4:end)=[tr.time(end);tr.gradient(1);tr.gradient(end);tr.perf(1);tr.perf(end)]
    training_results.net4 = tr;
    %NET5
    net5.trainParam.epochs=epochs_start;  % set the number of epochs for the training 
    [net5,tr]=train(net5,p,t);   % train the networks with batch
    a51=sim(net5,p); %a21=sim(net2,p);  % simulate the networks with the input vector p

    net5.trainParam.epochs=epochs_middle;
    [net5,tr]=train(net5,p,t);
    a52=sim(net5,p);

    net5.trainParam.epochs=epochs_end;
    [net5,tr]=train(net5,p,t);
    a53=sim(net5,p);
    results_matrix(5,4:end)=[tr.time(end);tr.gradient(1);tr.gradient(end);tr.perf(1);tr.perf(end)]
    training_results.net5 = tr;
    %NET6
    net6.trainParam.epochs=epochs_start;  % set the number of epochs for the training 
    [net6,tr]=train(net6,p,t);   % train the networks with batch
    a61=sim(net6,p); %a21=sim(net2,p);  % simulate the networks with the input vector p

    net6.trainParam.epochs=epochs_middle;
    [net6,tr]=train(net6,p,t);
    a62=sim(net6,p);

    net6.trainParam.epochs=epochs_end;
    [net6,tr]=train(net6,p,t);
    a63=sim(net6,p);
    results_matrix(6,4:end)=[tr.time(end);tr.gradient(1);tr.gradient(end);tr.perf(1);tr.perf(end)]
    training_results.net6 = tr;




    %PLOTS 1-2
    figure
    subplot(3,3,1);
    plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
    title(strcat(int2str(epochs_start),' epoch'));
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(3,3,2);
    [m,b,r_start1] = postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
    subplot(3,3,3);
    [m,b,r_start2]=postregm(cell2mat(a21),y);
    %
    subplot(3,3,4);
    plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
    title(strcat(int2str(epochs_middle),' epoch'));
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(3,3,5);
    [m,b,r_middle1] = postregm(cell2mat(a12),y);
    subplot(3,3,6);
    [m,b,r_middle2] =postregm(cell2mat(a22),y);
    %
    subplot(3,3,7);
    plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
    title(strcat(int2str(epochs_end),' epoch'));
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(3,3,8);
    [m,b,r_end1] =postregm(cell2mat(a13),y);
    subplot(3,3,9);
    [m,b,r_end2]=postregm(cell2mat(a23),y);

    %PLOTS 1-3
    figure
    subplot(3,3,1);
    plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a31),'g'); % plot the sine function and the output of the networks
    title(strcat(int2str(epochs_start),' epoch'));
    legend('target',algorithm1,algorithm3,'Location','north');
    subplot(3,3,2);
    [m,b,r_start1] = postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
    subplot(3,3,3);
    [m,b,r_start3]=postregm(cell2mat(a31),y);
    %
    subplot(3,3,4);
    plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a32),'g');
    title(strcat(int2str(epochs_middle),' epoch'));
    legend('target',algorithm1,algorithm3,'Location','north');
    subplot(3,3,5);
    [m,b,r_middle1] = postregm(cell2mat(a12),y);
    subplot(3,3,6);
    [m,b,r_middle3] =postregm(cell2mat(a32),y);
    %
    subplot(3,3,7);
    plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a33),'g');
    title(strcat(int2str(epochs_end),' epoch'));
    legend('target',algorithm1,algorithm3,'Location','north');
    subplot(3,3,8);
    [m,b,r_end1] =postregm(cell2mat(a13),y);
    subplot(3,3,9);
    [m,b,r_end3]=postregm(cell2mat(a33),y);

    %PLOTS 1-4
    figure
    subplot(3,3,1);
    plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a41),'g'); % plot the sine function and the output of the networks
    title(strcat(int2str(epochs_start),' epoch'));
    legend('target',algorithm1,algorithm4,'Location','north');
    subplot(3,3,2);
    [m,b,r_start1] = postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
    subplot(3,3,3);
    [m,b,r_start4]=postregm(cell2mat(a41),y);
    %
    subplot(3,3,4);
    plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a42),'g');
    title(strcat(int2str(epochs_middle),' epoch'));
    legend('target',algorithm1,algorithm4,'Location','north');
    subplot(3,3,5);
    [m,b,r_middle1] = postregm(cell2mat(a12),y);
    subplot(3,3,6);
    [m,b,r_middle4] =postregm(cell2mat(a42),y);
    %
    subplot(3,3,7);
    plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a43),'g');
    title(strcat(int2str(epochs_end),' epoch'));
    legend('target',algorithm1,algorithm4,'Location','north');
    subplot(3,3,8);
    [m,b,r_end1] =postregm(cell2mat(a13),y);
    subplot(3,3,9);
    [m,b,r_end4]=postregm(cell2mat(a43),y);


    %PLOTS 1-5
    figure
    subplot(3,3,1);
    plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a51),'g'); % plot the sine function and the output of the networks
    title(strcat(int2str(epochs_start),' epoch'));
    legend('target',algorithm1,algorithm5,'Location','north');
    subplot(3,3,2);
    [m,b,r_start1] = postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
    subplot(3,3,3);
    [m,b,r_start5]=postregm(cell2mat(a51),y);
    %
    subplot(3,3,4);
    plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a52),'g');
    title(strcat(int2str(epochs_middle),' epoch'));
    legend('target',algorithm1,algorithm5,'Location','north');
    subplot(3,3,5);
    [m,b,r_middle1] = postregm(cell2mat(a12),y);
    subplot(3,3,6);
    [m,b,r_middle5] =postregm(cell2mat(a52),y);
    %
    subplot(3,3,7);
    plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a53),'g');
    title(strcat(int2str(epochs_end),' epoch'));
    legend('target',algorithm1,algorithm5,'Location','north');
    subplot(3,3,8);
    [m,b,r_end1] =postregm(cell2mat(a13),y);
    subplot(3,3,9);
    [m,b,r_end5]=postregm(cell2mat(a53),y);



    %PLOTS 1-6
    figure
    subplot(3,3,1);
    plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a61),'g'); % plot the sine function and the output of the networks
    title(strcat(int2str(epochs_start),' epoch'));
    legend('target',algorithm1,algorithm6,'Location','north');
    subplot(3,3,2);
    [m,b,r_start1] = postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
    subplot(3,3,3);
    [m,b,r_start6]=postregm(cell2mat(a61),y);
    %
    subplot(3,3,4);
    plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a62),'g');
    title(strcat(int2str(epochs_middle),' epoch'));
    legend('target',algorithm1,algorithm6,'Location','north');
    subplot(3,3,5);
    [m,b,r_middle1] = postregm(cell2mat(a12),y);
    subplot(3,3,6);
    [m,b,r_middle6] =postregm(cell2mat(a62),y);
    %
    subplot(3,3,7);
    plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a63),'g');
    title(strcat(int2str(epochs_end),' epoch'));
    legend('target',algorithm1,algorithm6,'Location','north');
    subplot(3,3,8);
    [m,b,r_end1] =postregm(cell2mat(a13),y);
    subplot(3,3,9);
    [m,b,r_end6]=postregm(cell2mat(a63),y);



    results_matrix(1,1:3)=[r_start1;r_middle1;r_end1];
    results_matrix(2,1:3)=[r_start2;r_middle2;r_end2];
    results_matrix(3,1:3)=[r_start3;r_middle3;r_end3];
    results_matrix(4,1:3)=[r_start4;r_middle4;r_end4];
    results_matrix(5,1:3)=[r_start5;r_middle5;r_end5];
    results_matrix(6,1:3)=[r_start6;r_middle6;r_end6];



    save(strcat('matrix_results_',int2str(neurons),'_',int2str(attempt),'.mat'),'results_matrix')
    save(strcat('training_results_',int2str(neurons),'_',int2str(attempt),'.mat'),'training_results')

end
