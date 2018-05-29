clc;clear;close all;
pathTrainbr = 'RESULTS_nw/trainbr/'
pathOtherModels = 'RESULTS_nw/tansig/'
for neurons = ["5","10","20","30", "40", "50", "100", "200"]
    neurons= char(neurons)
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_1.mat'))
    matrix1 = results_matrix;
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_1.mat'))
    matrix1 = [matrix1;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_2.mat'))
    matrix2 = results_matrix;
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_2.mat'))
    matrix2 = [matrix2;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_3.mat'))
    matrix3 = results_matrix;
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_3.mat'))
    matrix3 = [matrix3;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_4.mat'))
    matrix4 = results_matrix;
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_4.mat'))
    matrix4 = [matrix4;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_5.mat'))
    matrix5 = results_matrix
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_5.mat'))
    matrix5 = [matrix5;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_6.mat'))
    matrix6 = results_matrix
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_6.mat'))
    matrix6 = [matrix6;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_7.mat'))
    matrix7 = results_matrix
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_7.mat'))
    matrix7 = [matrix7;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_8.mat'))
    matrix8 = results_matrix
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_8.mat'))
    matrix8 = [matrix8;results_matrix(1,:)];
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_9.mat'))
    matrix9 = results_matrix
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_9.mat'))
    matrix9 = [matrix9;results_matrix(1,:)];
    
    
    load(strcat(pathOtherModels,'matrix_results_',neurons,'_10.mat'))
    matrix10 = results_matrix
    load(strcat(pathTrainbr,'matrix_results_',neurons,'_10.mat'))
    matrix10 = [matrix10;results_matrix(1,:)];
    
    matrixAvg = (matrix1+matrix2+matrix3+matrix4+matrix5+matrix6+matrix7+matrix8+matrix9+matrix10)/10;
    save(strcat('RESULTS_nw/union/matrix_AVG_',(neurons),'.mat'),'matrixAvg')
end

%% Paint results
trainingAlgorithms = ["traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm","trainbr"]
load(strcat('RESULTS_nw/union/matrix_AVG_5.mat'))
matrix5 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_10.mat'))
matrix10 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_20.mat'))
matrix20 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_30.mat'))
matrix30 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_40.mat'))
matrix40 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_50.mat'))
matrix50 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_100.mat'))
matrix100 = matrixAvg
load(strcat('RESULTS_nw/union/matrix_AVG_200.mat'))
matrix200 = matrixAvg

for row_data =1:7
    % 1=MSE_train, 2=MSE_test ....mse_train,mse_test,r_train,r_test,tr.time(end),tr.epoch(end)
    close all
    name_row = ["MSE_training",  "MSE_val","MSE_test", "R_training", "R_test", "Training_time(s)", "Epochs"]

    alg = 1
    col1 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];


    alg = 2
    col2 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];

    alg = 3
    col3 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];

    alg = 4
    col4 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];

    alg = 5
    col5 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];

    alg = 6
    col6 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];
    
    alg = 7
    col7 = [matrix5(alg,row_data);matrix10(alg,row_data);matrix20(alg,row_data);matrix30(alg,row_data);matrix40(alg,row_data);
        matrix50(alg,row_data);matrix100(alg,row_data);matrix200(alg,row_data)];



    name = {'5';'10';'20';'30';'40';'50';'100';'200'};
    data = [col1  col2  col3 col4 col5 col6 col7];
    figure
    hb = bar(data, 'grouped')
    % set(hb(1), 'FaceColor','r')
    % set(hb(2), 'FaceColor','b')
    % set(hb(3), 'FaceColor','g')
    set(gca,'xticklabel',name)
    title(strcat('neurons vs ',name_row(row_data)));
    xlabel('neurons')
    ylabel(name_row(row_data))
    %legend("traingd", "traingda","traincgf","traincgp", "trainbfg", "trainlm", "trainbr")
    set(hb(1), 'FaceColor',[0.401 0.2 0.4901])
    set(hb(2), 'FaceColor',[0.5922    0.1961    0.5843])
    set(hb(3), 'FaceColor',[ 0.8471    0.1412    0.6431])
    set(hb(4), 'FaceColor',[0.9843    0.9686    0.1608])
    set(hb(5), 'FaceColor',[1.0000    0.6941         0])
    set(hb(6), 'FaceColor',[1.0000    0.9451    0.5686])
    set(hb(7), 'FaceColor','k')
    if(strcmp(name_row(row_data),"Training_time(s)"))
        ylim([0 60])
        set(gca, 'YScale', 'log')
    elseif(strcmp(name_row(row_data),"R_training")||strcmp(name_row(row_data),"R_test"))
        ylim([0 1])
    else
        set(gca, 'YScale', 'log')
    end
    
    grid on
    saveas(gcf,strcat('RESULTS_nw/union/',name_row(row_data),'.jpg'))
    savefig(strcat('RESULTS_nw/union/',name_row(row_data),'.fig'))
   
end
