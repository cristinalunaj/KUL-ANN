%% Test hipfield digits
clc; clear; close all
M = zeros(11,11);

load digits; clear size
[N, dim]=size(X);
maxx=max(max(X));

%Values must be +1 or -1
X(X==0)=-1;
%-------------------------------------------------------------------------

%Attractors of the Hopfield network

zero = X(1,:); %to visualize: digit=reshape(X(1,:),15, 16)'; -> imshow(digit); 
one = X(21,:);
two = X(41,:);
three = X(61,:);
four = X(81,:);
five = X(101,:);
six = X(121,:);
seven = X(141,:);
eight = X(161,:);
nine = X(181,:);

for attempt = 1:1000
    [Y,Yn, X] = hopdigit_v2(5,500);
    attempt
    close all
    [rows, cols]= size(Yn);
    for row=1:rows
        if(isequal(Yn(row,:),zero))
            M(row,1)=M(row,1)+1;
        elseif(isequal(Yn(row,:),one))
             M(row,2)=M(row,2)+1;
        elseif(isequal(Yn(row,:),two))
             M(row,3)=M(row,3)+1;
        elseif(isequal(Yn(row,:),three))
             M(row,4)=M(row,4)+1;
        elseif(isequal(Yn(row,:),four))
             M(row,5)=M(row,5)+1;
        elseif(isequal(Yn(row,:),five))
             M(row,6)=M(row,6)+1;
        elseif(isequal(Yn(row,:),six))
             M(row,7)=M(row,7)+1;
        elseif(isequal(Yn(row,:),seven))
             M(row,8)=M(row,8)+1;
        elseif(isequal(Yn(row,:),eight))
             M(row,9)=M(row,9)+1;
        elseif(isequal(Yn(row,:),nine))
             M(row,10)=M(row,10)+1;
        else
             M(row,11)=M(row,11)+1;
        end
    end
    
end

%% PLOT
imagesc(M);
colormap(flipud(gray));
textStrings = num2str(M(:));       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  % Remove any space padding
[x, y] = meshgrid(1:11);  % Create x and y coordinates for the strings
hStrings = text(x(:), y(:), textStrings(:), ...  % Plot the strings
                'HorizontalAlignment', 'center');
midValue = mean(get(gca, 'CLim'));  % Get the middle value of the color range
textColors = repmat(M(:) > midValue, 1, 3);  % Choose white or black for the
                                               %   text color of the strings so
                                               %   they can be easily seen over
                                               %   the background color
set(hStrings, {'Color'}, num2cell(textColors, 2));  % Change the text colors

set(gca, 'XTick', 1:11, ...                             % Change the axes tick marks
         'XTickLabel', {'0','1', '2', '3', '4', '5', '6','7','8','9','spurious'}, ...  %   and tick labels
         'YTick', 1:11, ...
         'YTickLabel', {'0','1', '2', '3', '4', '5', '6','7','8','9','spurious'}, ...
         'TickLength', [0 0]);
xlabel('Predicted digit');
ylabel('Real digit');
     