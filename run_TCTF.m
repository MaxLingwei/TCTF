clear all;
close all;
img_origin = double(imread('124084.jpg'));
p = 0.7;
sample_index = rand(size(img_origin)) < p;
%load('sample_index.mat');
img_sample = img_origin .* sample_index;
Omega = find(sample_index > 0);
r0 = [30, 30, 30];
X = [];
Y = [];
C = [];
r = [];
[X, Y, C, r] = TCTF(img_sample, Omega, r0, img_origin);

M = zeros(size(img_origin));
[n1, n2, n3] = size(img_sample);
for i = 1:n3
    get_X = X(1:n1, 1:r(i), i);
    get_Y = Y(1:r(i), 1:n2, i);
    M(:, :, i) = get_X * get_Y;
end
result = real(ifft(M, [], 3));


psnr = PSNR(n1, n2, n3, img_origin, C)


pic = cat(2, uint8(img_origin), uint8(img_sample), uint8(C));
figure; montage(pic);