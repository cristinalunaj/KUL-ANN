clc, clear;
x=0:0.05:3*pi; t=sin(x.^2)+0.1*randn(1,size(x,2));
x_test =0:0.02:2*pi; y_test = sin(x_test.^2)+0.1*randn(1,size(x_test,2));%+0.01*randn(1,size(x,2))   

net = newff(x,t,30,{},'trainbr');
net = train(net,x,t);
net.trainParam.epochs = 1000;
y=sim(net, x);

pred=sim(net,x_test);
plot(x,t,'*'), hold, plot(x,y,'r-')
%test
figure;
plot(x_test,y_test,'bx',x_test,pred,'r');
title('TEST');
legend('target','Bayes','Location','north');