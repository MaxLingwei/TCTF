function [X, Y, C, r] = TCTF(M, Omega, r0, img_origin)
min_rank = [25, 5, 5];
max_iter = 150;
epsilon = 1e-5;
normdata = norm(M(Omega));
value_pixel = M(Omega);
r1 = r0;
[n1, n2, n3] = size(M);
rhat = max(r0);


[Xhat, Yhat] = initial_2(r0, img_origin, Omega)

rse_set = [];
%load('hat.mat');
for iter = 1:max_iter
    iter
    Xhat_k = Xhat;
    Yhat_k = Yhat;
    if iter > 1
        Chat_k = Chat;
    end
    
    for k = 1:n3
        get_Xhat = Xhat{k};
        get_Yhat = Yhat{k};
        %C_temp = get_Xhat * get_Yhat;
        %Chat(1:size(C_temp, 1), 1:size(C_temp, 2), k) = C_temp;
        Chat(:, :, k) = get_Xhat * get_Yhat;
    end
    if iter == 1
        Chat_k = zeros(size(Chat));
    end
    
    C = real(ifft(Chat, [], 3));
    diff = norm(C(Omega) - M(Omega)) / normdata;
    if diff < 1e-8
        C(Omega) = M(Omega);
        break;
    end
    
    C(Omega) = value_pixel;
    
    rse_set = [rse_set, RSE(img_origin, C)];
    
    Chat = fft(C, [], 3);
    
    for k = 1:n3
        get_Yhat = Yhat{k};
        YtY = get_Yhat * get_Yhat';
        X_temp = Chat(:, :, k) * get_Yhat' * pinv(YtY);
        Xhat{k} = X_temp;
        
        get_Xhat = Xhat{k};
        XtX = get_Xhat' * get_Xhat;
        Y_temp = pinv(XtX) * get_Xhat' * Chat(:, :, k);
        Yhat{k} = Y_temp;
        
        
    end
    if sum(r1) > sum(min_rank)
        [Xhat, Yhat, r1] = estimation_rank_t(Xhat, Yhat, r1);
    end
    
%     diff = Chat - Chat_k;
%     if max(abs(diff(:))) < epsilon && cal_diff(Xhat, Xhat_k) < epsilon && cal_diff(Yhat, Yhat_k) < epsilon
%         break;
%     end
%     if abs(max(max(max(Xhat - Xhat_k)))) < epsilon && abs(max(max(max(Yhat - Yhat_k)))) < epsilon && abs(max(max(max(Chat - Chat_k)))) < epsilon
%         break;
%     end
end
X = Xhat;
Y = Yhat;
r = r1;

function [Xhat, Yhat, r1] = estimation_rank_t(Xhat, Yhat, r0)
min_rank = [25, 5, 5];
n3 = length(Xhat);
multi_rank = r0;
eigen_recorder = [];
eigen_val = [];
for i = 1:n3
    XtX = Xhat{i}' * Xhat{i};
    eigen_XtX = real(eig(XtX));
    eigen_val = [eigen_val; eigen_XtX];
    eigen_recorder = [eigen_recorder; length(eigen_XtX)];
end
eigen_recorder = cumsum([1; eigen_recorder]);
eigen_val_sort = sort(eigen_val, 'descend');
quotients = eigen_val_sort(1:end - 1) ./ eigen_val_sort(2:end);
[lambda_tk, t_k] = max(quotients);
tau_k = (sum(r0) - 1) * lambda_tk / (sum(quotients) - lambda_tk);

Xhat_decrease = Xhat;
Yhat_decrease = Yhat;
if tau_k >= 10
    eig_val_cum = cumsum(eigen_val_sort);
    sk_set = find(eig_val_cum / sum(eigen_val_sort) >= 0.95);
    sk = sk_set(1);
    
    
    eig_sk_nk = eigen_val_sort(sk + 1:end);
    for i = 1:n3
        eigen_i = sort(eigen_val(eigen_recorder(i): eigen_recorder(i + 1) - 1));
        mk_i = max(find(eigen_i < eig_sk_nk(1)));
        if length(mk_i) == 0
            mk_i = 0;
        end
        
        multi_rank(i) = max(multi_rank(i) - mk_i, min_rank(i));
        [U, S, V] = svd(Xhat{i} * Yhat{i});
        temp_US = U(:, 1:multi_rank(i)) * S(1:multi_rank(i), 1:multi_rank(i));
        Xhat_decrease{i} = temp_US;
        temp_V = V(:, 1:multi_rank(i))';
        Yhat_decrease{i} = temp_V;
    end
    
end
Xhat = Xhat_decrease;
Yhat = Yhat_decrease;
r1 = multi_rank;

function [Xhat, Yhat] = initial_Time_Domain(r0, M)
[n1, n2, n3] = size(M);
for i = 1:n3
    X(:, 1:r0(i), i) = randn(n1, r0(i));
    Y(1:r0(i), :, i) = randn(r0(i), n2);
end
Xhat = fft(X, [], 3);
Yhat = fft(Y, [], 3);

function [Xhat, Yhat] = initial_Frequency_Domain(r0, M)
rhat = max(r0);
[n1, n2, n3] = size(M);
Xhat = ones(n1, rhat, n3);
Yhat = ones(rhat, n2, n3);
rand_real = randn(n1, r0(1));
rand_img = randn(n1, r0(1));
Xhat(:, 1:r0(1), 1) = randn(n1, r0(1));
Xhat(:, 1:r0(2), 2) = rand_real + rand_img * i;
Xhat(:, 1:r0(3), 3) = rand_real - rand_img * i;

rand_real = randn(r0(1), n2);
rand_img = rand(r0(1), n2);
Yhat(1:r0(1), :, 1) = randn(r0(2), n2);
Yhat(1:r0(2), :, 2) = rand_real + rand_img * i;
Yhat(1:r0(3), :, 3) = rand_real - rand_img * i;

function [test_eigen_val] = test_func_geteig(tensor)
[n1, n2, n3] = size(tensor);
eigen = [];
for k = 1:n3
    eigen = [eigen; eig(tensor(:, :, k) * tensor(:, :, k)')];
end
test_eigen_val = eigen;

function [X, Y] = initial_2(r0, Origin, Omega)
TC=randn(size(Origin));
TC(Omega)=Origin(Omega);
TC=fft(TC,[],3);
for i = 1:length(r0)
    [Uk,Sigmak,Vk]=svds(TC(:,:,i),r0(i));
    X{i} = Uk*Sigmak;
    Y{i} = Vk';
end

function [max_diff] = cal_diff(A, B)
temp = [];
c = [];
for i = 1:length(A)
    c = A{i} - B{i};
    temp = [temp; c(:)];
end
max_diff = max(abs(temp));
