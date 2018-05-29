clear
clc
close all

%Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load iris.dat
X = iris(:,1:end-1);
true_labels = iris(:,end); 
class1 = iris(1:50,:);
class2 = iris(51:100,:);
class3 = iris(101:end,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Training the SOM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_length = 3;
y_length = 1;
gridsize=[y_length x_length];% CREO QUE PARA VUESTRO CASO SERÍAN [2 1] O [1 2] SI QUEIRES SÓLO 2 NEURONAS O 'CLUSTERS'
net = newsom(X',gridsize,'hextop','mandist');

% plot the data distribution with the prototypes of the untrained network
figure;
colormap(jet)
scatter3(class1(:,1),class1(:,2),class1(:,5),'r', 'filled');
hold on
scatter3(class2(:,1),class2(:,2),class2(:,5),'g','filled');
hold on
scatter3(class3(:,1),class3(:,2),class3(:,5),'b','filled');
grid on
legend('class1','class2','class3');
xlabel('Sepal length')
ylabel('Sepal width')

figure;
colormap(jet)
scatter3(class1(:,3),class1(:,4),class1(:,5),'r', 'filled');
hold on
scatter3(class2(:,3),class2(:,4),class2(:,5),'g','filled');
hold on
scatter3(class3(:,3),class3(:,4),class3(:,5),'b','filled');
grid on 
legend('class1','class2','class3');
xlabel('Petal length')
ylabel('Petal width')
%figure
%plotsom(net.iw{1},net.layers{1}.distances)
% hold off



net.trainParam.epochs = 200;
[net,tr] = train(net,X');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Assigning examples to clusters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outputs = sim(net,X');
[~,assignment]  =  max(outputs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Compare clusters with true labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ARI=RandIndex(assignment,true_labels);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
