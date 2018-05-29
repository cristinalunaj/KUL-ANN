function paintLetters_v2(letterMatrix)

digitConc =[];
for i = 1:size(letterMatrix,2)
digit = reshape(letterMatrix(:,i),5,7)'; 
digitConc = [digitConc digit];
end
imshow(digitConc)