clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%
%DATA%%%%%%%%
%generation of examples and targets
%x=0:0.05:3*pi; y=sin(x.^2);
x=0:0.05:3*pi; y=sin(x.^2)%+0.01*randn(1,size(x,2));
x_test =0:0.02:2*pi; y_test = sin(x_test.^2)%+0.01*randn(1,size(x,2))   
x_test2 = 3*pi:0.05:6*pi; y_test2=sin(x_test2.^2)
algorithm1 = 'traingd';
algorithm2 = 'trainbfg'; %'trainbfg';'trainlm'; 'traincgf'
neurons = 200;
epochs_middle = 15;
epochs_end = 50;
R_middle_epochs_alg1=[]; 
R_middle_epochs_alg2=[];
R_end_epoch_alg1 = []; 
R_end_epoch_alg2 = []; 
n_repetitions = 1;
%noisy data: 

%%%%%%%%%%%%%%%%%
p=con2seq(x); t=con2seq(y); % convert the data to a useful format
test_p = con2seq(x_test); test_t = con2seq(y_test)
test_p2 = con2seq(x_test2); test_t2 = con2seq(y_test2)
for i=1:n_repetitions

    %creation of networks
    net1=feedforwardnet(neurons,algorithm1);
    net2=feedforwardnet(neurons,algorithm2);
    
    net1 = configure(net1,p,t);
    net2 = configure(net2,p,t);
    
%     net1.iw{1,1}=rand(neurons,1);  %set the same weights and biases for the networks 
%     net1.lw{2,1}=rand(1,neurons);
%     net1.b{1}=rands(neurons,1);
%     net1.b{2}=rands(1,1);
    
    save net1
    
    net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
    net2.lw{2,1}=net1.lw{2,1}
    net2.b{1}=net1.b{1}
    net2.b{2}=net1.b{2}
 

    %training and simulation
    net1.trainParam.epochs=1;  % set the number of epochs for the training 
    net2.trainParam.epochs=1;
    net1=train(net1,p,t);   % train the networks with batch
    net2=train(net2,p,t);
    a11=sim(net1,p); a21=sim(net2,p);  % simulate the networks with the input vector p

    net1.trainParam.epochs=epochs_middle;
    net2.trainParam.epochs=epochs_middle;
    net1=train(net1,p,t);
    net2=train(net2,p,t);
    a12=sim(net1,p); a22=sim(net2,p);

    net1.trainParam.epochs=epochs_end;
    net2.trainParam.epochs=epochs_end;
    net1=train(net1,p,t);
    net2=train(net2,p,t);
    a13=sim(net1,p); a23=sim(net2,p);

    %plots
    figure
    subplot(3,3,1);
    plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
    title('1 epoch');
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(3,3,2);
    postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
    subplot(3,3,3);
    postregm(cell2mat(a21),y);
    %
    subplot(3,3,4);
    plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
    title(strcat(int2str(epochs_middle),' epochs'));
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(3,3,5);
    [m,b,R_m1] = postregm(cell2mat(a12),y);
    subplot(3,3,6);
    [m,b,R_m2] = postregm(cell2mat(a22),y);
    R_middle_epochs_alg1=[R_middle_epochs_alg1,abs(R_m1)]
    R_middle_epochs_alg2=[R_middle_epochs_alg2,abs(R_m2)]
    %
    subplot(3,3,7);
    plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
    title(strcat(int2str(epochs_end),' epochs'));
    legend('target',algorithm1,algorithm2','Location','north');
    subplot(3,3,8);
    [m,b,R_e1] = postregm(cell2mat(a13),y);
    subplot(3,3,9);
    [m,b,R_e2] = postregm(cell2mat(a23),y);
    
    R_end_epoch_alg1=[R_end_epoch_alg1,abs(R_e1)]
    R_end_epoch_alg2=[R_end_epoch_alg2,abs(R_e2)]
    
    
    % 500 & 1000 epochs
 
    net1.trainParam.epochs=500;
    net2.trainParam.epochs=500;
    net1=train(net1,p,t);
    net2=train(net2,p,t);
    a14=sim(net1,p); a24=sim(net2,p);
    
  
    net1.trainParam.epochs=1000;
    net2.trainParam.epochs=1000;
    net1=train(net1,p,t);
    net2=train(net2,p,t);
    a15=sim(net1,p); a25=sim(net2,p);
    
     %plots
    figure
    subplot(2,3,1);
    plot(x,y,'bx',x,cell2mat(a14),'r',x,cell2mat(a24),'g'); % plot the sine function and the output of the networks
    title('500 epoch');
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(2,3,2);
    postregm(cell2mat(a14),y); % perform a linear regression analysis and plot the result
    subplot(2,3,3);
    postregm(cell2mat(a24),y);
    %
    subplot(2,3,4);
    plot(x,y,'bx',x,cell2mat(a15),'r',x,cell2mat(a25),'g');
    title('1.000 epochs');
    legend('target',algorithm1,algorithm2,'Location','north');
    subplot(2,3,5);
    [m,b,R_m1] = postregm(cell2mat(a15),y);
    subplot(2,3,6);
    [m,b,R_m2] = postregm(cell2mat(a25),y);

  
    
    
    
    %TEST FOR END EPOCHS: 
    figure; 
    pred_alg1 = sim(net1,test_p);
    pred_alg2 = sim(net2,test_p);
    plot(x_test,y_test,'bx',x_test,cell2mat(pred_alg1),'r',x_test,cell2mat(pred_alg2),'g');
    title('TEST');
    legend('target',algorithm1,algorithm2','Location','north');
    figure;
    [m,b,R_m2] = postregm(cell2mat(pred_alg1),y_test);
    figure;
    [m,b,R_m2] = postregm(cell2mat(pred_alg2),y_test);
    
    figure; 
    pred2_alg1 = sim(net1,test_p2);
    pred2_alg2 = sim(net2,test_p2);
    plot(x_test2,y_test2,'bx',x_test2,cell2mat(pred2_alg1),'r',x_test2,cell2mat(pred2_alg2),'g');
    title('TEST2');
    legend('target',algorithm1,algorithm2','Location','north');
    figure;
    [m,b,R_m2] = postregm(cell2mat(pred2_alg1),y_test2);
    figure;
    [m,b,R_m2] = postregm(cell2mat(pred2_alg2),y_test2);
    
    
    
    
end
% avg_R_middle_alg1 = sum(R_middle_epochs_alg1)/size(R_middle_epochs_alg1,1)
% avg_R_middle_alg2 = sum(R_middle_epochs_alg2)/size(R_middle_epochs_alg2,1)
% 
% 
% avg_R_end_alg1 = sum(R_end_epoch_alg1)/size(R_end_epoch_alg1,1)
% avg_R_end_alg2 = sum(R_end_epoch_alg2)/size(R_end_epoch_alg2,1)
