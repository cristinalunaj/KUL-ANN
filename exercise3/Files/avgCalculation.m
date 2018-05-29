clc
clear
close all
%neurons = '20';
for epochs = ["100","200","300"]%["10","50","100","200","300", "400", "500","1000"]
    epochs= char(epochs)
    
    load(strcat('matrix_results_',epochs,'_1.mat'))
    matrix1 = results_matrix;
    load(strcat('matrix_results_',epochs,'_2.mat'))
    matrix2 = results_matrix;
    load(strcat('matrix_results_',epochs,'_3.mat'))
    matrix3 = results_matrix;
    load(strcat('matrix_results_',epochs,'_4.mat'))
    matrix4 = results_matrix;
    load(strcat('matrix_results_',epochs,'_5.mat'))
    matrix5 = results_matrix
    load(strcat('matrix_results_',epochs,'_6.mat'))
    matrix6 = results_matrix
    load(strcat('matrix_results_',epochs,'_7.mat'))
    matrix7 = results_matrix
    load(strcat('matrix_results_',epochs,'_8.mat'))
    matrix8 = results_matrix
    load(strcat('matrix_results_',epochs,'_9.mat'))
    matrix9 = results_matrix
    load(strcat('matrix_results_',epochs,'_10.mat'))
    matrix10 = results_matrix



    matrixAvg = (matrix1+matrix2+matrix3+matrix4+matrix5+matrix6+matrix7+matrix8+matrix9+matrix10)/10;
    save(strcat('matrix_AVG_',(epochs),'.mat'),'matrixAvg')
end

%% Paint results
%trainingAlgorithms = ["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm"]
% load(strcat('matrix_AVG_10.mat'))
% matrix10 = matrixAvg
% load(strcat('matrix_AVG_50.mat'))
% matrix50 = matrixAvg
load(strcat('matrix_AVG_100.mat'))
matrix100 = matrixAvg
load(strcat('matrix_AVG_200.mat'))
matrix200 = matrixAvg
load(strcat('matrix_AVG_300.mat'))
matrix300 = matrixAvg
% load(strcat('matrix_AVG_400.mat'))
% matrix400 = matrixAvg
% load(strcat('matrix_AVG_500.mat'))
% matrix500 = matrixAvg
% load(strcat('matrix_AVG_1000.mat'))
% matrix1000 = matrixAvg

for row_data =1:3
    % 1=MSE_train, 2=MSE_test ....mse_train,mse_test,r_train,r_test,tr.time(end),tr.epoch(end)
    close all
    name_row = ["ARI","Training-time(s)", "Epochs"]
    alg = 1
    col1 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 2
    col2 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 3
    col3 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 4
    col4 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 5
    col5 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 6
    col6 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 7
    col7 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 1
    col1 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 8
    col8 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 9
    col9 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 10
    col10 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
    alg = 11
    col11 = [matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data)];
%     alg = 1
%     col1 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 2
%     col2 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 3
%     col3 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 4
%     col4 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 5
%     col5 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%    alg = 6
%     col6 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 7
%     col7 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 8
%     col8 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 9
%     col9 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 10
%     col10 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 11
%     col11 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
%     alg = 12
%     col12 = [matrix10(alg,row_data);matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data);matrix300(alg,row_data);
%         matrix400(alg,row_data);matrix500(alg,row_data);matrix1000(alg,row_data)];
    
    
    name = {'100';'200';'300'}%{'10';'50';'100';'200';'300';'400';'500';'1000'};
    data = [col1  col2  col3 col4 col5 col6, col7 col8 col9 col10 col11];
    figure
    hb = bar(data, 'grouped')
    % set(hb(1), 'FaceColor','r')
    % set(hb(2), 'FaceColor','b')
    % set(hb(3), 'FaceColor','g')
    set(gca,'xticklabel',name)
    title(strcat('epochs vs ',name_row(row_data)));
    xlabel('epochs')
    ylabel(name_row(row_data))
    %legend("tanh-tanh", "tanh-sig","tanh-linear","sig-tanh", "sig-sig", "sig-linear")
    set(hb(1), 'FaceColor',[0.401 0.2 0.4901])
    set(hb(2), 'FaceColor',[0.5922    0.1961    0.5843])
    set(hb(3), 'FaceColor',[ 0.8471    0.1412    0.6431])
    set(hb(4), 'FaceColor',[0.9843    0.9686    0.1608])
    set(hb(5), 'FaceColor',[1.0000    0.6941         0])
    set(hb(6), 'FaceColor',[1.0000    0.9451    0.5686])
%     set(hb(7), 'FaceColor','k')
%     if(strcmp(name_row(row_data),"MSE-training")||strcmp(name_row(row_data),"MSE-test"))
%         ylim([0 1])
%     end
    
    grid on
    saveas(gcf,strcat(name_row(row_data),'.jpg'))
    savefig(strcat(name_row(row_data),'.fig'))
   
end