function [ w ] = linearperceptron( K,d,nu,tran )
%UNT�TLED Summary of this function goes here
%   Detailed explanation goes here
% Inputs:
% K =     number of classes
% d  =    number of inputs
% nu =    learning factor

r = zeros(K,size(tran,1));
for i=1:K
    r(i,30*(i-1)+1:30*i) = 1;
end

% Training Phase
w = -0.01 + 0.02.*rand(K,d+1);
y = zeros(1,K);
for EPOCH=1:1000
    idx = randperm(size(tran,1));
    for t = 1:size(tran,1)
        o = w*tran(idx(t),1:d+1)';
        for i=1:K
            y(i) = exp(o(i))/sum(exp(o));
        end
        for i=1:K
            for j=1:d+1
                w(i,j) = w(i,j) + nu.*(r(i,idx(t))-y(i)).*tran(idx(t),j);
            end
        end
    end
end

end

