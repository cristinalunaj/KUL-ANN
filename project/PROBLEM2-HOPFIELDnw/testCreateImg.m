% test createImg
clc
clear; close all
alphabet = prprob();

amplificationFactor = 2; 
imgSize = 16;
T = []; 
for i= 1:15
    letterVector = alphabet(:,i);
    imgAsVector = createImg(letterVector, amplificationFactor, imgSize)
    imgAsVector(imgAsVector==0) = -1;
    T = [T,imgAsVector];
end

digitConc =[];
for i = 1:size(T,2)
digit = reshape(T(:,i),16,16)'; 
digitConc = [digitConc digit'];
end
imshow(digitConc)