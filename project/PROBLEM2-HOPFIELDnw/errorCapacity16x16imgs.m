%% testing capacity 16X16 IMGS
clc;
clear all; 
close all
num_iterations = [500, 750, 1000];
errorVector = zeros(3, 35);
errorVectorLetter = zeros(3, 35);
errorVectorAux = []
errorVectorLetterAux = []
normalAlp = 0
index = 1
for attempt=1:10
    index=1
    errorVectorAux = []; 
    errorVectorLetterAux = [];
    errorVector = zeros(3, 35);
    errorVectorLetter = zeros(3, 35);
    for num_iter = num_iterations
        for num_letters = 1:35
            [error, letterMistaken] = capacityCalculator(num_letters,num_iter, normalAlp);
            errorVectorAux = [errorVectorAux, error];
            errorVectorLetterAux = [errorVectorLetterAux,letterMistaken];
        end
        errorVector(index, :) = errorVectorAux;
        errorVectorLetter(index,:) = errorVectorLetterAux;
        errorVectorAux = []; 
        errorVectorLetterAux = [];
        index = index+1
    end
    save(strcat('RESULTS16x16/results_errorVecP3_',num2str(attempt), '.mat'), 'errorVector')
    save(strcat('RESULTS16x16/results_errorLetterP3_',num2str(attempt), '.mat'), 'errorVectorLetter')
end

% x = 1:35
% plot(x,errorVector)
% figure
% plot(x,errorVectorLetter)
