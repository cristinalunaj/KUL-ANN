%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%
%ARE THE ATTRACTORS STATES ALWAYS THOSE STORED IN THE NW AT THE CREATION?
%   No, sometimes there are some states that appear in the center so when
%   we initialize the points to some values in the center of 2 attractor
%   states, there is another attractor state in that position and the
%   point doesn't evolve and stay in that postion
clc;clear
T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
plot(T(1,:),T(2,:),'k*');  
hold on

P= [0 0 0 1 -1 0.3 -1;
    0 1 -1 0 0 0.7 1];

P = [0 0;
    0 1;
    0 -1;
    1 0;
    -1 0;
    -0.5 -0.5;
    -0.5 0.5;
    0.3 0.7;
    -1 1;
    ]'
n=9
Y = {};
%{[0,0],[0,1],[0,-1], [1,0],[-1,0]};
for i=1:n
    a={P(:,i)};%{rands(2,1)};  % generate an initial point 
    [y,Pf,Af] = net({50},{},a);   % simulation of the network for 50 timesteps  
    Y(i,:) = y; 
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
