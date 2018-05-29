%% LOAD APPAHBET
close all
clear
clc
alphabet = prprob();
alphabet(alphabet==0)=-1;
numAttractors = size(alphabet,1)
num_letters = 5;
Ttrain = alphabet;
T = alphabet(:,1:num_letters);
%Create network
net = newhop(Ttrain);
num_iter = 800;

%Check if digits are attractors
[Y,~,~] = sim(net,num_letters,[],T);
Y = Y';


figure
subplot(num_letters,3,1);
for i = 1:num_letters
letter = Y(i,:);
letter = reshape(letter,5,7)'; 

subplot(num_letters,3,((i-1)*3)+1);
imshow(letter)

if i == 1
    title('Attractors')
end
hold on
end


% Swipe 3 pixels random


X =  alphabet(:,1:num_letters);
X_distorted = X;
for i=1:num_letters;
    randomPxs = randperm(size(alphabet,1));
    for px = 1:3
        X_distorted(randomPxs(px),i)=X_distorted(randomPxs(px),i)*(-1);
    end
end


% %Show noisy digits:


subplot(num_letters,3,2);
for i = 1:num_letters
    letter = X_distorted(:,i);
    letter = reshape(letter,5,7)';
    %subplot(num_letters*3,2,i);
    subplot(num_letters,3,((i-1)*3)+2);
    imshow(letter)
if i == 1
    title('Noisy digits')
end
hold on
end



%------------------------------------------------------------------------

%See if the network can correct the corrupted digits 

num_steps = num_iter;

[Yn,~,~] = sim(net,{num_letters num_steps},{},X_distorted);
Yn = Yn{1,num_steps};
Yn = Yn'


subplot(num_letters,3,3);
for i = 1:num_letters
letter = Yn(i,:);
letter = reshape(letter,5,7)';
subplot(num_letters,3,((i-1)*3)+3);
imshow(letter)
if i == 1
    title('Reconstructed noisy digits')
end
hold on
end


