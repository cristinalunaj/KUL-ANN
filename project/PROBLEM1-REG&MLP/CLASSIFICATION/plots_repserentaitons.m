% plot features/labels
dataset = csvread('winequality-red_commaSeparated.csv',1);
%SELECTION OF THE CLASSES THAT WE ARE GONNA USE IN THE CLASSIFICATION TASK
[indexClassesRow,indexClassesCols]=find(dataset(:,end)==5 | dataset(:,end)==6);
binaryDS = dataset(indexClassesRow,:);

%RANDOMIZATION OF THE ROWS IN THE DS
numSamples = size(binaryDS,1);
load('permutation.mat');%randperm(numSamples);
binaryDS_random = binaryDS(permutation,:);

%SELECTION OF THE TEST SAMPLES, 98 OF C+ & 98 OF C-
test_set= [binaryDS_random(find(binaryDS_random(:,end)==5,98),:);binaryDS_random(find(binaryDS_random(:,end)==6,98),:)];
binaryDS_random(find(binaryDS_random(:,end)==5,98),:)=[];
binaryDS_random(find(binaryDS_random(:,end)==6,98),:)=[];


%CREATION OF VALIDATION AND TEST SETS
numSamples = size(binaryDS_random,1);
trainRatio = 0.83
valRatio = 0.17
testRatio = 0
[trainInd,valInd,testInd] = divideblock(numSamples,trainRatio,valRatio,testRatio);



train_set = binaryDS_random(trainInd,:);
X = train_set(:,1:end-1);
Y = train_set(:,end);
indexClass1 = find(Y==5);
otherClass = find(Y==6);



for feat = 1:size(X,2)

%     max = max(Xt(:,feat));
%     min = min(Xt(:,feat));
%     range = max-min;
    figure;
    hold on;
    histogram(X(indexClass1,feat),50)
    hold on;
    histogram(X(otherClass,feat),50)
    grid on
    title(strcat('whine DataSet - x',num2str(feat)))
    legend('C+','C-')
    xlabel(strcat('X',num2str(feat)))
%     saveas(gcf,strcat('plots/CLASS_feat',num2str(feat),'.jpg'))
%     savefig(strcat('plots/CLASS_feat',num2str(feat),'.fig'))

end

close all
labels = {'X1' 'X2' 'X3' 'X4' 'X5' 'X6' 'X7' 'X8' 'X9' 'X10' 'X11' 'Y'};
figure
data = [X(:,1) X(:,2) X(:,3) X(:,4) X(:,5) X(:,6) X(:,7) X(:,8) X(:,9) X(:,10) X(:,11) Y];

[h,ax] = plotmatrix(data); 

for i = 1:12                                      % label the plots
  xlabel(ax(12,i), labels{i})
  ylabel(ax(i,1), labels{i})
end

