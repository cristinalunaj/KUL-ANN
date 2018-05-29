clear
clc
close all

%Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load iris.dat
X = iris(:,1:end-1);
true_labels = iris(:,end); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Training the SOM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_epochs = [100,200,300];
% x_length = [1 3];
% y_length = [3 1];
xy_length = [[2 1];[1 2];[3 1];[1 3];[4 1];[1 4];[5 1];[1 5];[2 2];[3 2];[2 3]];
%grids = {'gridtop', 'hextop', 'randtop'};
%distances = {'dist','boxdist', 'linkdist','mandist'}; 
n_repetitions = 10
results_matrix = zeros(11, 3);


ARI_vector = zeros(24,1); 
index_parameter=1;
for attempt = 1:n_repetitions
    for epochs =num_epochs
        for gridSize = 1:11 
        %for gridsValues = 1:3
            %for distancesValues = 1:4
                %grid_name = (grids(gridsValues));
                %dist_name = distances(distancesValues);
                %net1 = newsom(X',xy_length,grid_name{1,1},dist_name{1,1});
                grid = xy_length(gridSize,:)
                net1 = newsom(X',grid,'hextop','linkdist');
                net1.trainParam.epochs = epochs;
                [net1,tr] = train(net1,X');
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                % Assigning examples to clusters
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                outputs = sim(net1,X');
                [~,assignment]  =  max(outputs);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                %Compare clusters with true labels
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ARI=RandIndex(assignment,true_labels);
                results_matrix(index_parameter,:) = [ARI,tr.time(end),tr.epoch(end)];
                index_parameter=index_parameter+1;
            end
        %end
        save(strcat('matrix_results_',int2str(epochs),'_',int2str(attempt),'.mat'),'results_matrix')
        index_parameter=1;
    end
end

% plot(x_length, ARI_vector(1,:),'r')
% hold on
% plot(x_length, ARI_vector(2,:),'g')
% hold on
% plot(x_length, ARI_vector(3,:),'b')
% hold on 
% legend('top1: gridtop & dist', 'top2: hextop & boxdist', 'top3: randtop & linkdist');
% xlabel('x length'); 
% ylabel('ARI');
%[sorted_x, index] = sort(ARI_vector,'ascend');


