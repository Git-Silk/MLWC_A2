clc; clear; close all;
rng(41)

N = 64;
cp_len = 16;
num_symbols = 2;

snr_list = 0:5:25;
num_test = 5000;
num_channels = 500;

pilot_bits_64 = randi([0 1], 2*N, 1);
Xp_64 = (2*pilot_bits_64(1:N)-1) + 1j*(2*pilot_bits_64(N+1:end)-1);
Xp_64 = Xp_64 / sqrt(2);

pilot_idx_8 = 1:8:N;
Xp_8 = zeros(N,1);
pilot_bits_8 = randi([0 1], 2*length(pilot_idx_8), 1);
Xp_temp = (2*pilot_bits_8(1:length(pilot_idx_8))-1) + 1j*(2*pilot_bits_8(length(pilot_idx_8)+1:end)-1);
Xp_temp = Xp_temp / sqrt(2);
Xp_8(pilot_idx_8) = Xp_temp;

cfg = winner2.wimparset;
cfg.CenterFrequency = 2.6e9;

BSAA = winner2.AntennaArray('UCA',1,0.5);
MSAA = winner2.AntennaArray('ULA',1,0.5);

layout = winner2.layoutparset(2,{1},1,[BSAA MSAA]);
layout.Stations(1).Pos = [0;0;30];
layout.Stations(2).Pos = [200;0;1.5];
layout.ScenarioVector = 11;

H_bank = cell(num_channels,1);
delay_bank = cell(num_channels,1);

for i = 1:num_channels
    cfg.RandomSeed = randi(1e6);
    [H, delays] = winner2.wim(cfg, layout);
    H_bank{i} = H;
    delay_bank{i} = delays;
end

H_samples = zeros(N,num_channels);

for i = 1:num_channels
    H = H_bank{i}; delays = delay_bank{i};
    h_full = squeeze(H{1}); h_full = h_full(:,1);

    delay_samples = round(delays / cfg.DelaySamplingInterval);
    h = zeros(N,1);

    for k = 1:length(h_full)
        d = delay_samples(k);
        if d <= 16
            h(d+1) = h(d+1) + h_full(k);
        end
    end

    H_samples(:,i) = fft(h,N);
end

R_HH = (H_samples*H_samples')/num_channels;
R_HH = R_HH / trace(R_HH) * N;

[BER_LS_64, BER_MMSE_64, X_test_64, Y_test_64] = simulate_OFDM(Xp_64, [], true, snr_list, num_test, N, cp_len, num_channels,H_bank, delay_bank, cfg, R_HH);

[BER_LS_8, BER_MMSE_8, X_test_8, Y_test_8] = simulate_OFDM(Xp_8, pilot_idx_8, true, snr_list, num_test, N, cp_len, num_channels,H_bank, delay_bank, cfg, R_HH);

[BER_LS_noCP, BER_MMSE_noCP, X_test_noCP, Y_test_noCP] = simulate_OFDM(Xp_64, [], false, snr_list, num_test, N, cp_len, num_channels,H_bank, delay_bank, cfg, R_HH);


save('test_all_data.mat', ...
    'snr_list', ...
    'X_test_64','Y_test_64','BER_LS_64','BER_MMSE_64', ...
    'X_test_8','Y_test_8','BER_LS_8','BER_MMSE_8', ...
    'X_test_noCP','Y_test_noCP','BER_LS_noCP','BER_MMSE_noCP', ...
    '-v7.3');

figure;
semilogy(snr_list, BER_LS_64, '-o','LineWidth',2); hold on;
semilogy(snr_list, BER_MMSE_64, '-s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('64 Pilots (With CP)');
legend('LS','MMSE');

figure;
semilogy(snr_list, BER_LS_8, '-o','LineWidth',2); hold on;
semilogy(snr_list, BER_MMSE_8, '-s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('8 Pilots (With CP)');
legend('LS','MMSE');

figure;
semilogy(snr_list, BER_LS_64, '-o','LineWidth',2); hold on;
semilogy(snr_list, BER_MMSE_64, '-s','LineWidth',2);
semilogy(snr_list, BER_LS_noCP, '--o','LineWidth',2);
semilogy(snr_list, BER_MMSE_noCP, '--s','LineWidth',2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('CP vs No CP (64 Pilots)');
legend('LS CP','MMSE CP','LS NoCP','MMSE NoCP');

%% Helper functions

function [BER_LS, BER_MMSE, X_test, Y_test] = simulate_OFDM(Xp, pilot_idx, use_cp, snr_list, num_test,N, cp_len, num_channels, H_bank, delay_bank, cfg, R_HH)

BER_LS = zeros(size(snr_list));
BER_MMSE = zeros(size(snr_list));

X_test = cell(length(snr_list),1);
Y_test = cell(length(snr_list),1);

for s = 1:length(snr_list)

    snr_dB = snr_list(s);

    err_ls = 0; err_mmse = 0;

    X_temp = zeros(num_test, 4*N);
    Y_temp = zeros(num_test, 2*N);

    parfor i = 1:num_test

        bits = randi([0 1], 2*N, 1);
        Xd = (2*bits(1:N)-1) + 1j*(2*bits(N+1:end)-1);
        Xd = Xd/sqrt(2);

        if isempty(pilot_idx)
            X = [Xp, Xd];
        else
            Xp_full = zeros(N,1);
            Xp_full(pilot_idx) = Xp(pilot_idx);
            Xd(pilot_idx) = 0;
            X = [Xp_full + Xd, Xd];
        end

        
        x_time = ifft(X,N);

        if use_cp
            x_cp = [x_time(end-cp_len+1:end,:); x_time];
        else
            x_cp = x_time;
        end

        x_serial = x_cp(:);

       
        idx = randi(num_channels);
        H = H_bank{idx}; delays = delay_bank{idx};

        h_full = squeeze(H{1}); h_full = h_full(:,1);
        delay_samples = round(delays / cfg.DelaySamplingInterval);

        h = zeros(N,1);
        for k = 1:length(h_full)
            d = delay_samples(k);
            if d <= 16
                h(d+1) = h(d+1) + h_full(k);
            end
        end

        y = conv(x_serial,h);
        y = y(1:length(x_serial));

        
        signal_power = mean(abs(y).^2);
        noise_var = signal_power / (10^(snr_dB/10));
        noise = sqrt(noise_var/2)*(randn(size(y)) + 1j*randn(size(y)));
        y = y + noise;

      
        if use_cp
            y_blocks = reshape(y, N+cp_len, []);
            y_noCP = y_blocks(cp_len+1:end,:);
        else
            y_noCP = reshape(y, N, []);
        end

        Y = fft(y_noCP,N);

        Yp = Y(:,1);
        Yd = Y(:,2);

        
        X_temp(i,:) = [real(Yp); imag(Yp); real(Yd); imag(Yd)].';
        Y_temp(i,:) = bits.';

        
        if isempty(pilot_idx)
            H_LS = Yp ./ Xp;
        else
            H_LS = zeros(N,1);
            H_LS(pilot_idx) = Yp(pilot_idx) ./ Xp(pilot_idx);
            H_LS = interp1(pilot_idx, H_LS(pilot_idx), 1:N, 'linear','extrap').';
        end

        X_est_LS = Yd ./ H_LS;
        bits_ls = [real(X_est_LS)>0; imag(X_est_LS)>0];
        err_ls = err_ls + sum(bits ~= bits_ls);

        
        H_MMSE = R_HH * ((R_HH + noise_var*eye(N)) \ H_LS);
        X_est_MMSE = Yd ./ H_MMSE;

        bits_mmse = [real(X_est_MMSE)>0; imag(X_est_MMSE)>0];
        err_mmse = err_mmse + sum(bits ~= bits_mmse);

    end

    BER_LS(s) = err_ls/(num_test*2*N);
    BER_MMSE(s) = err_mmse/(num_test*2*N);

    X_test{s} = X_temp;
    Y_test{s} = Y_temp;

end
end