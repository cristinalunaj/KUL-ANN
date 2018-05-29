%% HOPFIELD NW
% eXERCISE1

% -DO THERE EXIST OTHER ATTRACTOR STATES BESIDE THE ONES THAT HAVE BEEN
% PROGRAMMED INTO THE NW?? 
%   yes, there is at least one [-1 1], whcih is the complementary of the
%   attractor [1 -1]

% - HOW LONG DOES IT TYPICALLY TAKE TO REACH THE ATTRACTOR STATES?? 
%     around 7-8 steps  

clc; clear;
T = [1 1;
    -1 -1;
    1 -1]';
%T= [1 1;-1 -1];

num_steps =20; 
%[Y,Pf,Af]=net({num_steps}, {}, T);

net = newhop(T);
W= net.LW{1,1}

a = [1 -1]';
[y1,Pf,Af] = net({20},{},a);


a = [-1 1]';
[y2,Pf,Af] = net({20},{},a);

a = [0.5 0.5]';
[y3,Pf,Af] = net({20},{},a);

a = [-1 0.5]';
[y4,Pf,Af] = net({20},{},a);


a = [-0.5 -0.5]';
[y5,Pf,Af] = net({20},{},a);

% axis([-1 1 -1 1 -1 1])
% gca.box = 'on';
% axis manual;
% hold on;
% plot3(T(1,:),T(2,:),T(3,:),'r*')
% title('Hopfield Network State Space')
% xlabel('a(1)');
% ylabel('a(2)');
% zlabel('a(3)');
% view([37.5 30]);



a = {rands(3,1)};
[y,Pf,Af] = net({1},{},T);
W= net.LW{1,1}
b = net.b{1}
[y,Pf,Af] = net({1},{},T);
W= net.LW{1,1}
[y,Pf,Af] = net({1},{},T);
W= net.LW{1,1}




record = [cell2mat(a) cell2mat(y)];
start = cell2mat(a);
hold on
plot3(start(1,1),start(2,1),start(3,1),'bx', ...
   record(1,:),record(2,:),record(3,:),'b',record(1,end),record(2,end),record(3,end),'gx')

repetitions = 25; 
Y = {};
steps = zeros(1,repetitions)

color = 'rgbmy';
for i = 1:repetitions
   a = {rand(3,1)};
   [y,Pf,Af] = net({1 num_steps},{},a);
   Y(i,:) = y; 
   for j=1:num_steps
       if((sum(y{1,j}==T(:,1))==3)|| (sum(y{1,j}==T(:,2))==3))
           steps(1,i) = j;
           break
       end
   end
   record = [cell2mat(a) cell2mat(y)];
   start = cell2mat(a);
   plot3(start(1,1),start(2,1),start(3,1),'kx', ...
      record(1,:),record(2,:),record(3,:),color(rem(i,5)+1), record(1,end),record(2,end),record(3,end),'gx')
end

% average = sum(steps)/size(steps,2)