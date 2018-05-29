x = [-2:.1:2];
p = 26;
N=35;
stdDev = p/N;
norm = normpdf(x,0,stdDev);
figure;
plot(x,norm)
perror = sum(norm(31:end))