clear all;
close all;
img_origin = double(imread('178054.jpg'));
img_origin = double(img_origin) / 255.0;

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
    get_X = X{i};
    get_Y = Y{i};
    M(:, :, i) = get_X * get_Y;
end
result = real(ifft(M, [], 3));


psnr = PSNR(n1, n2, n3, img_origin, C)


pic = cat(2, img_origin, img_sample, C);
figure; montage(pic);