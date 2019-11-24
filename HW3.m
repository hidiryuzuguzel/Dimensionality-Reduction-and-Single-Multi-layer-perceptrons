% HW3: Dimension reduction and single/multi layer perceptrons
% Author: Hidir Yuzuguzel
clear all;clc;

Data = load('iris.txt');

tran_data = [Data(1:30,1:5);Data(51:80,1:5);Data(101:130,1:5)];
test_data = [Data(31:50,1:5);Data(81:100,1:5);Data(131:150,1:5)];

tran = [ones(size(tran_data,1),1),tran_data];
test = [ones(size(test_data,1),1),test_data];
%%
sigmoid = @(x) 1./(1+exp(-x));
figure('Name','Data visualization','NumberTitle','off');
col = {'bx', 'ro', 'k+'};
%% -------------------Linear Perceptron------------------
w = linearperceptron(3,4,0.1,tran); % training phase

% Testing Phase
for i=1:size(test,1)
    res = w*test(i,1:5)';
    [maxVal maxIndex] = max(res);
    test(i,7) = maxIndex;
end

for i=1:size(tran,1)
    res2 = w*tran(i,1:5)';
    [maxVal2 maxIndex2] = max(res2);
    tran(i,7) = maxIndex2;
end  

% compute confusion matrix for linear perceptron (training)
Conf_perceptron_tran = zeros(3,3);
for i=1:size(tran_data,1)
    Conf_perceptron_tran(tran(i,6),tran(i,7)) = ...
        Conf_perceptron_tran(tran(i,6),tran(i,7))+1;
end
Conf_perceptron_tran

% compute confusion matrix for linear perceptron (test)
Conf_perceptron_test = zeros(3,3);
for i=1:size(test_data,1)
    Conf_perceptron_test(test(i,6),test(i,7)) = ...
        Conf_perceptron_test(test(i,6),test(i,7))+1;
end
Conf_perceptron_test


%% -------------------- Multilayer Perceptron (MLP)--------------------

[w,v,z] = MLP (3,2,4,0.1,tran);  % training phase

zhat = ones(2+1,1);
% Testing Phase
for i=1:size(test,1)
    zhat(2:end) = sigmoid(w*test(i,1:5)');
    o = v'*zhat;
    for k=1:3
        res(k) = exp(o(k))/sum(exp(o));
    end
    [maxVal maxIndex] = max(res);
    test(i,8) = maxIndex;
end

myres = w*test(:,1:5)';
for i=1:size(tran,1)
    zhat(2:end) = sigmoid(w*tran(i,1:5)');
    o = v'*zhat;
    for k=1:3
        res2(k) = exp(o(k))/sum(exp(o));
    end
    [maxVal2 maxIndex2] = max(res2);
    tran(i,8) = maxIndex2;
end 

myres2 = w*tran(:,1:5)';
% compute confusion matrix for MLP (training)
Conf_mlp_tran = zeros(3,3);
for i=1:size(tran_data,1)
    Conf_mlp_tran(tran(i,6),tran(i,8)) = ...
        Conf_mlp_tran(tran(i,6),tran(i,8))+1;
end
Conf_mlp_tran

% compute confusion matrix for MLP (test)
Conf_mlp_test = zeros(3,3);
for i=1:size(test_data,1)
    Conf_mlp_test(test(i,6),test(i,8)) = ...
        Conf_mlp_test(test(i,6),test(i,8))+1;
end
Conf_mlp_test

subplot(321)
for i=1:3
    plot(myres2(1,30*(i-1)+1:30*i),myres2(2,30*(i-1)+1:30*i),col{i});
    hold on;
end
title('MLP training data')

subplot(322)
for i=1:3
    plot(myres(1,20*(i-1)+1:20*i),myres(2,20*(i-1)+1:20*i),col{i});
    hold on;
end
title('MLP test data')


%% ---------------------- PCA -------------------------------------
dim = 2;
[eigvec eigval] = eig (cov(Data(:,1:4)));
z_pca = (eigvec(:,end-dim+1:end)' * Data(:,1:4)')';
%plot(z(1:50,1),'x',z(51:100,2),'t',z(101:150,:),'o');

z_pca = [z_pca,Data(:,5)];
tran_z_pca = [z_pca(1:30,1:3);z_pca(51:80,1:3);z_pca(101:130,1:3)];
test_z_pca = [z_pca(31:50,1:3);z_pca(81:100,1:3);z_pca(131:150,1:3)];

tran_z_pca = [ones(size(tran_z_pca,1),1),tran_z_pca];
test_z_pca = [ones(size(test_z_pca,1),1),test_z_pca];

w_pca = linearperceptron(3,2,0.1,tran_z_pca);

% Testing Phase
for i=1:size(test_z_pca,1)
    res = w_pca*test_z_pca(i,1:3)';
    [maxVal maxIndex] = max(res);
    test_z_pca(i,5) = maxIndex;
end

for i=1:size(tran_z_pca,1)
    res2 = w_pca*tran_z_pca(i,1:3)';
    [maxVal2 maxIndex2] = max(res2);
    tran_z_pca(i,5) = maxIndex2;
end

% compute confusion matrix for PCA+linear perceptron (training)
Conf_pca_tran = zeros(3,3);
for i=1:size(tran_z_pca,1)
    Conf_pca_tran(tran_z_pca(i,4),tran_z_pca(i,5)) = ...
        Conf_pca_tran(tran_z_pca(i,4),tran_z_pca(i,5))+1;
end
Conf_pca_tran

% compute confusion matrix for PCA+linear perceptron (test)
Conf_pca_test = zeros(3,3);
for i=1:size(test_z_pca ,1)
    Conf_pca_test(test_z_pca (i,4),test_z_pca (i,5)) = ...
        Conf_pca_test(test_z_pca (i,4),test_z_pca (i,5))+1;
end
Conf_pca_test

subplot(323)
for i=1:3
    plot(tran_z_pca(30*(i-1)+1:30*i,2),tran_z_pca(30*(i-1)+1:30*i,3),col{i});
    hold on;
end
title('PCA training data')

subplot(324)
for i=1:3
    plot(test_z_pca(20*(i-1)+1:20*i,2),test_z_pca(20*(i-1)+1:20*i,3),col{i});
    hold on;
end
title('PCA test data')
%% ---------------------- LDA -------------------------------------
mi = zeros(3,4);
% for i=1:3
%     mi(i,:) = mean(Data(50*(i-1)+1:50*i,1:4));
% %     Si(i,:) = sum(bsxfun(@times,bsxfun(@minus,Data(50*(i-1)+1:50*i,1:4),mi(i,:)),...
% %         bsxfun(@times,Data(50*(i-1)+1:50*i,1:4),mi(i,:))));
%     Si(:,:,i) = bsxfun(@minus,Data(50*(i-1)+1:50*i,1:4),mi(i,:))'*...
%         bsxfun(@minus,Data(50*(i-1)+1:50*i,1:4),mi(i,:));
% end

mi_1 = mean(Data(1:50,1:4));
mi_2 = mean(Data(51:100,1:4));
mi_3 = mean(Data(101:150,1:4));

mi = [mi_1;mi_2;mi_3];

Si_1 = bsxfun(@minus,Data(1:50,1:4),mi_1)'*...
      bsxfun(@minus,Data(1:50,1:4),mi_1);
Si_2 = bsxfun(@minus,Data(51:100,1:4),mi_2)'*...
      bsxfun(@minus,Data(51:100,1:4),mi_2);
Si_3 = bsxfun(@minus,Data(101:150,1:4),mi_3)'*...
      bsxfun(@minus,Data(101:150,1:4),mi_3);

Sw = Si_1 + Si_2 + Si_3; % Within-class scatter matrix

m = (mi_1+mi_2+mi_3)/3;

Sb = 50*bsxfun(@minus,mi,m)'*bsxfun(@minus,mi,m); % between-class scatter matrix

dim = 2;
J = Sw\Sb;
[eigvec eigval] = eig (J(:,1:4));
z_lda = (eigvec(:,1:dim)' * Data(:,1:4)')';

z_lda = [z_lda,Data(:,5)];
tran_z_lda = [z_lda(1:30,1:3);z_lda(51:80,1:3);z_lda(101:130,1:3)];
test_z_lda = [z_lda(31:50,1:3);z_lda(81:100,1:3);z_lda(131:150,1:3)];

tran_z_lda = [ones(size(tran_z_lda,1),1),tran_z_lda];
test_z_lda = [ones(size(test_z_lda,1),1),test_z_lda];

w_lda = linearperceptron(3,2,0.1,tran_z_lda);

% Testing Phase
for i=1:size(test_z_lda,1)
    res = w_lda*test_z_lda(i,1:3)';
    [maxVal maxIndex] = max(res);
    test_z_lda(i,5) = maxIndex;
end

for i=1:size(tran_z_lda,1)
    res2 = w_lda*tran_z_lda(i,1:3)';
    [maxVal2 maxIndex2] = max(res2);
    tran_z_lda(i,5) = maxIndex2;
end

% compute confusion matrix for LDA+linear perceptron (training)
Conf_lda_tran = zeros(3,3);
for i=1:size(tran_z_lda,1)
    Conf_lda_tran(tran_z_lda(i,4),tran_z_lda(i,5)) = ...
        Conf_lda_tran(tran_z_lda(i,4),tran_z_lda(i,5))+1;
end
Conf_lda_tran

% compute confusion matrix for LDA+linear perceptron (test)
Conf_lda_test = zeros(3,3);
for i=1:size(test_z_lda ,1)
    Conf_lda_test(test_z_lda (i,4),test_z_lda (i,5)) = ...
        Conf_lda_test(test_z_lda (i,4),test_z_lda (i,5))+1;
end
Conf_lda_test

subplot(325)
for i=1:3
    plot(tran_z_lda(30*(i-1)+1:30*i,2),tran_z_lda(30*(i-1)+1:30*i,3),col{i});
    hold on;
end
title('LDA training data')

subplot(326)
for i=1:3
    plot(test_z_lda(20*(i-1)+1:20*i,2),test_z_lda(20*(i-1)+1:20*i,3),col{i});
    hold on;
end
title('LDA test data')