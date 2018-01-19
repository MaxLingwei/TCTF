function [peak_signal_noise_ratio] = PSNR(n1, n2, n3, M, Chat)
peak_signal_noise_ratio = 10 * log10((n1 * n2 * n3 * max(max(max(M))) ^ 2) / sum(sum(sum((Chat - M) .^ 2))));