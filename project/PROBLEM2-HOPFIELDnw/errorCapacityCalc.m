%% testing capacity
clc;
clear all; 
close all
num_iterations = [500, 750, 1000];
errorVector = zeros(3, 25);
errorVectorLetter = zeros(3, 25);
errorVectorAux = []
errorVectorLetterAux = []
normalAlp = 1
index = 1
for attempt=1:10
    index=1
    errorVectorAux = []; 
    errorVectorLetterAux = [];
    errorVector = zeros(3, 25);
    errorVectorLetter = zeros(3, 25);
    for num_iter = num_iterations
        for num_letters = 1:25
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
    save(strcat('RESULTS/sinMias/results_errorVecP3_',num2str(attempt), '.mat'), 'errorVector')
    save(strcat('RESULTS/sinMias/results_errorLetterP3_',num2str(attempt), '.mat'), 'errorVectorLetter')
end

% x = 1:35
% plot(x,errorVector)
% figure
% plot(x,errorVectorLetter)
