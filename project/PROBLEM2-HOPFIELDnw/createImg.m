function [imgAsVector] = createImg(letterVector, amplificationFactor, imgSize)

    letter = reshape(letterVector,5,7)';
    B = imresize(letter,amplificationFactor);
    B1 = B;
    B1(B1<=0.5)=0;
    B1(B1>0.5)=1;
   
    newImg = zeros(imgSize,imgSize);
    newImg(2:15,4:13)=B1;
    %imshow(newImg);
    imgAsVector = reshape(newImg,imgSize*imgSize,1);

end