%% LOADING CAPACITY CALCULATION 

function [error,errorLetter] = capacityCalculator(num_letters,num_iter, normalAlp)
alphabet = prprob();
%alphabet = alphabet(:,11:end)
    if(normalAlp==1)
        alphabet(alphabet==0)=-1;
        %attractors
        T = alphabet(:,1:num_letters);
        numPx=35
    else
        numPx=256
        amplificationFactor = 2; 
        imgSize = 16;
        T = []; 
        for i= 1:num_letters
            letterVector = alphabet(:,i);
            imgAsVector = createImg(letterVector, amplificationFactor, imgSize);
            imgAsVector(imgAsVector==0) = -1;
            T = [T imgAsVector];
        end
    end
    %Create network
    net = newhop(T);
    %Check if digits are attractors
    [Y,~,~] = sim(net,num_letters,[],T);
    Y = Y';


%     figure
%     for i = 1:num_letters
%     letter = Y(i,:);
%     letter = reshape(letter,5,7)'; 
% 
%     subplot(num_letters,1,i);
%     imshow(letter)
% 
%     if i == 1
%         title('Attractors')
%     end
%     hold on
%     end


    % Swipe 3 pixels random
    X_distorted = T;
    for i=1:num_letters;
        randomPxs = randperm(size(T,1));
        for px = 1:3
            X_distorted(randomPxs(px),i)=X_distorted(randomPxs(px),i)*(-1);
        end
    end


    % %Show noisy digits:
%     figure
%     for i = 1:num_letters
%         letter = X_distorted(:,i);
%         letter = reshape(letter,5,7)';
%         subplot(num_letters,1,i);
%         imshow(letter)
%     if i == 1
%         title('Noisy digits')
%     end
%     hold on
%     end



    %------------------------------------------------------------------------

    %See if the network can correct the corrupted digits 

    num_steps = num_iter;

    [Yn,~,~] = sim(net,{num_letters num_steps},{},X_distorted);
    Yn = Yn{1,num_steps};
    Yn = Yn';
    roundTargets = [1 -1];
    incorrectPx = 0;
    incorrectLetter = 0;
    totalPx = numPx*num_letters;
    T=T';
    [rows,cols] = size(Yn);
    for r=1:rows
        yrRounded = interp1(roundTargets,roundTargets,Yn(r,:),'nearest');
        if(sum(sum(yrRounded-T(r,:)))~=0)
            incorrectLetter = incorrectLetter+1;
        end
            
        for c=1:cols
            if(yrRounded(c)==T(r,c))
                %igual
            else
                incorrectPx=incorrectPx+1;
        end
    end
    error = (incorrectPx/totalPx)*100;
    errorLetter = (incorrectLetter/num_letters)*100;
        
%     figure
%     for i = 1:num_letters
%     letter = Yn(i,:);
%     letter = reshape(letter,5,7)';
%     subplot(num_letters,1,i);
%     imshow(letter)
%     if i == 1
%         title('Reconstructed noisy digits')
%     end
%     hold on
%     end
close all
end

