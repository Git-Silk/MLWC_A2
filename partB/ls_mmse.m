clear; clc; close all;
rng(42);


%  SYSTEM PARAMETERS
M      = 4;
Nx     = 16;  Ny = 16;
N      = Nx * Ny;           % 256 RIS elements
K      = 1;
Nsub   = 64;
Np     = 64;
CP     = 16;
bps    = 2;                 % QPSK: 2 bits/symbol
L_eff  = CP;                % Effective CIR taps = 16

SNR_dB = 0:2:30;
n_SNR  = length(SNR_dB);
Nmc    = 500;               % Trials per SNR point
N_test_total = Nmc * n_SNR;

pwr_dly = exp(-(0:L_eff-1)' / L_eff);
pwr_dly = pwr_dly / sum(pwr_dly);

X_pilot = ones(Np, 1);

r_col   = fft([pwr_dly; zeros(Nsub-L_eff, 1)], Nsub);
[J, I]  = meshgrid(1:Nsub, 1:Nsub);
idx_mat = mod(I-J, Nsub) + 1;
R_HH    = r_col(idx_mat);
R_HH    = (R_HH + R_HH') / 2;         
fprintf('Done.\n\n');

SER_LS   = zeros(n_SNR, 1);
SER_MMSE = zeros(n_SNR, 1);

X_test     = zeros(256, N_test_total, 'single');
Y_test     = zeros(128, N_test_total, 'single');
SNR_labels = zeros(1,   N_test_total, 'single');

for si = 1:n_SNR

    SNR    = 10^(SNR_dB(si) / 10);
    sigma2 = 1 / SNR;

    W_MMSE = R_HH / (R_HH + sigma2 * eye(Nsub));

    err_LS   = 0;
    err_MMSE = 0;
    total    = 0;

    col_offset = (si-1) * Nmc;     % Column offset into flat test arrays

    for mc = 1:Nmc

        h_cir  = sqrt(pwr_dly/2) .* (randn(L_eff,1) + 1j*randn(L_eff,1));
        H_freq = fft([h_cir; zeros(Nsub-L_eff,1)], Nsub);   % (64x1)

        n_pilot = sqrt(sigma2/2) * (randn(Nsub,1) + 1j*randn(Nsub,1));
        Y_pilot = H_freq .* X_pilot + n_pilot;

        H_LS       = Y_pilot ./ X_pilot;

        H_MMSE_est = W_MMSE * H_LS;

        bits_tx = randi([0 1], Nsub*bps, 1);                % 128 bits
        sym_tx  = qpsk_mod(bits_tx);                         % (64x1)

        n_data = sqrt(sigma2/2) * (randn(Nsub,1) + 1j*randn(Nsub,1));
        Y_data = H_freq .* sym_tx + n_data;                  % (64x1)

        sym_LS    = Y_data ./ H_LS;
        bits_LS   = qpsk_demod(sym_LS);

        sym_MMSE  = Y_data ./ H_MMSE_est;
        bits_MMSE = qpsk_demod(sym_MMSE);

        ref_m  = reshape(bits_tx,   bps, []);   % (2 x 64)
        ls_m   = reshape(bits_LS,   bps, []);
        mmse_m = reshape(bits_MMSE, bps, []);

        err_LS   = err_LS   + sum(any(ref_m ~= ls_m,   1));
        err_MMSE = err_MMSE + sum(any(ref_m ~= mmse_m, 1));
        total    = total + Nsub;

        col = col_offset + mc;
        X_test(:, col) = single([ real(Y_pilot); imag(Y_pilot); ...
                                   real(Y_data);  imag(Y_data)  ]);
        Y_test(:, col) = single(bits_tx);
        SNR_labels(col) = single(si);     

    end  

    SER_LS(si)   = err_LS   / total;
    SER_MMSE(si) = err_MMSE / total;
    fprintf('  %5d dB    %12.6f    %12.6f\n', SNR_dB(si), SER_LS(si), SER_MMSE(si));

end

SNR_dB_vec = single(SNR_dB);      

save('test_data_ris.mat', ...
     'X_test',     ...   % (256  x N_test_total)
     'Y_test',     ...   % (128  x N_test_total)
     'SNR_labels', ...   % (1    x N_test_total) SNR index 1..n_SNR per sample
     'SNR_dB_vec', ...   % (1    x n_SNR)        dB values of SNR axis
     'SER_LS',     ...   % (n_SNR x 1)
     'SER_MMSE',   ...   % (n_SNR x 1)
     '-v7.3');           % HDF5 – required for Python h5py
fprintf('Done.\n');
fprintf('  X_test     : [%d x %d]\n', size(X_test,1),    size(X_test,2));
fprintf('  Y_test     : [%d x %d]\n', size(Y_test,1),    size(Y_test,2));
fprintf('  SNR_labels : %d unique levels  (1..%d)\n', n_SNR, n_SNR);
fprintf('  SNR_dB_vec : [');
fprintf('%g ', SNR_dB_vec); fprintf(']\n');

%  FIGURE: SER vs SNR  (LS vs MMSE)
figure('Name','Part B Task2 – LS vs MMSE','Position',[100 100 720 540]);

semilogy(SNR_dB, SER_LS,   'r-o', 'LineWidth',2.0, 'MarkerSize',8, ...
         'MarkerFaceColor','r', 'DisplayName','LS Estimation');
hold on;
semilogy(SNR_dB, SER_MMSE, 'b-s', 'LineWidth',2.0, 'MarkerSize',8, ...
         'MarkerFaceColor','b', 'DisplayName','MMSE Estimation');

grid on; grid minor;
xlabel('SNR (dB)',                'FontSize',13);
ylabel('Symbol Error Rate (SER)','FontSize',13);
title({'RIS-Assisted MU-MISO OFDM  –  Part B Task 2', ...
       'LS vs MMSE Channel Estimation  (Pilots=64, CP=16)'}, 'FontSize',13);
legend('Location','southwest','FontSize',12);
xlim([SNR_dB(1), SNR_dB(end)]);
ylim([1e-4, 1]);
set(gca,'FontSize',12);


fprintf('\n[Done] test_data_ris.mat\n');

%  LOCAL FUNCTIONS
function sym = qpsk_mod(bits)
    bits = reshape(bits, 2, []);
    I    = 1 - 2*bits(1,:);
    Q    = 1 - 2*bits(2,:);
    sym  = (I + 1j*Q).' / sqrt(2);
end

function bits = qpsk_demod(sym)
    sym  = sym(:) * sqrt(2);
    b1   = double(real(sym) < 0);
    b2   = double(imag(sym) < 0);
    bits = reshape([b1'; b2'], [], 1);
end
