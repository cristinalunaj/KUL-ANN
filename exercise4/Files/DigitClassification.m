clear all
first_layer_neurons = [100, 150, 200, 300, 400]
epochs_first_layer = [400,500,600,800]
epochs_second_later = [100,200,300,400]
acc_matrix_autoEncoders = []
acc_matrix_MLP = []


for n=1:10
    for neurons1 = first_layer_neurons
        for epochs1 = epochs_first_layer
            for epochs2 = epochs_second_later

                %clear all
                close all
                nntraintool('close');
                nnet.guis.closeAllViews();
               

                % Neural networks have weights randomly initialized before training.
                % Therefore the results from training are different each time. To avoid
                % this behavior, explicitly set the random number generator seed.
                %rng('default')


                % Load the training data into memory
                %[xTrainImages, tTrain] = digittrain_dataset;
                load('digittrain_dataset');

                % Layer 1
                hiddenSize1 = neurons1;
                autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
                    'MaxEpochs',epochs1, ...
                    'L2WeightRegularization',0.004, ...
                    'SparsityRegularization',4, ...
                    'SparsityProportion',0.15, ...
                    'ScaleData', false);

                figure;
                plotWeights(autoenc1);
                feat1 = encode(autoenc1,xTrainImages);

                % Layer 2
                hiddenSize2 = round(neurons1/2);
                autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
                    'MaxEpochs',epochs2, ...
                    'L2WeightRegularization',0.002, ...
                    'SparsityRegularization',4, ...
                    'SparsityProportion',0.1, ...
                    'ScaleData', false);

                feat2 = encode(autoenc2,feat1);

                % Layer 3
                softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',epochs1);


                % Deep Net
                deepnet = stack(autoenc1,autoenc2,softnet);


                % Test deep net
                imageWidth = 28;
                imageHeight = 28;
                inputSize = imageWidth*imageHeight;
                %[xTestImages, tTest] = digittest_dataset;
                load('digittest_dataset');
                xTest = zeros(inputSize,numel(xTestImages));
                for i = 1:numel(xTestImages)
                    xTest(:,i) = xTestImages{i}(:);
                end
                y = deepnet(xTest);
                figure;
                plotconfusion(tTest,y);
                classAcc=100*(1-confusion(tTest,y))
                new_encoder = [neurons1, epochs1,epochs2,classAcc];
                acc_matrix_autoEncoders=[acc_matrix_autoEncoders;new_encoder];

                % Test fine-tuned deep net
                xTrain = zeros(inputSize,numel(xTrainImages));
                for i = 1:numel(xTrainImages)
                    xTrain(:,i) = xTrainImages{i}(:);
                end
                deepnet = train(deepnet,xTrain,tTrain);
                y = deepnet(xTest);
                figure;
                plotconfusion(tTest,y);
                saveas(gcf, 'CMAES.fig');
                classAcc=100*(1-confusion(tTest,y))
                view(deepnet)

                %Compare with normal neural network (1 hidden layers)
                net = patternnet(neurons1);
                net=train(net,xTrain,tTrain);
                y=net(xTest);
                plotconfusion(tTest,y);
                saveas(gcf, 'CM_mlp1.fig');
                classAcc1=100*(1-confusion(tTest,y))
                view(net)

                % %Compare with normal neural network (2 hidden layers)
                %
                net = patternnet([neurons1,round(neurons1/2)]);
                net=train(net,xTrain,tTrain);
                y=net(xTest);
                plotconfusion(tTest,y);
                saveas(gcf, 'CM_mlp2.fig');
                classAcc2=100*(1-confusion(tTest,y))
                view(net)



                % %Compare with normal neural network (3 hidden layers)
                net = patternnet([neurons1,round(neurons1/2),round(neurons1/3)]);
                net=train(net,xTrain,tTrain);
                y=net(xTest);
                plotconfusion(tTest,y);
                saveas(gcf, 'CM_mlp3.fig');
                classAcc3=100*(1-confusion(tTest,y))
                view(net)
                
                new_mlp = [neurons1, epochs1,epochs2,classAcc1,classAcc2,classAcc3];
                acc_matrix_MLP=[acc_matrix_MLP;new_mlp]
            end
        end
    end
end

save('acc_autoEncoders.mat', 'acc_matrix_autoEncoders');
save('acc_MLPs.mat', 'acc_matrix_MLP');
