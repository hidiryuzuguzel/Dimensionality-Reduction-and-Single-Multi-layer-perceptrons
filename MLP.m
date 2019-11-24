function [ w,v,z ] = MLP( K,H,d,nu,tran )
%UNT�TLED2 Summary of this function goes here
%   Detailed explanation goes here

r = zeros(K,size(tran,1));
for i=1:K
    r(i,30*(i-1)+1:30*i) = 1;
end
v = -0.01 + 0.02.*rand(K,H+1);
% w = -0.01 + 0.02.*rand(H+1,d+1);  % WRONG
w = -0.01 + 0.02.*rand(H,d+1);
z = ones(H+1,1);
y = zeros(K,1);
sigmoid = @(x) 1./(1+exp(-x));
EPOCH = 100;

% Training Phase
for ep=1:EPOCH
    idx = randperm(size(tran,1));
    for t = 1:size(tran,1)
        %% Forward-pass
%         z(2:H+1) = sigmoid(w(2:H+1,:)*tran(idx(t),1:d+1)');
        z(2:end) = sigmoid(w*tran(idx(t),1:d+1)');
        o = v'*z;
        
        for i=1:K
            y(i) = exp(o(i))/sum(exp(o));
        end
        %% Backward-pass
        dv = nu.*z*(r(:,idx(t))-y)';
        
%         dw = nu .* (((r(:,idx(t))-y)'*v(:,2:H+2-1))'.*z(2:2+H-1).*...
%             (1-z(2:2+H-1)))*tran(idx(t),1:d+1);
        dw = nu .* (((r(:,idx(t))-y)'*v)'.*z.*...
            (1-z))*tran(idx(t),1:d+1);
        
        v = v + dv;
%         w(2:2+H-1,:) = w(2:2+H-1,:) + dw;
        w = w + dw(2:end,:);
        
    end
end
end

