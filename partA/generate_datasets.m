clc; clear; close all;
rng(41)

N = 64;
cp_len = 16;
num_symbols = 2;

num_train = 500000;
snr_range = [0 30];

data = load('channel_bank.mat'); 

H_bank = data.H_bank;
delay_bank = data.delay_bank;
cfg = data.cfg;

num_channels = length(H_bank);

pilot_bits_64 = randi([0 1], 2*N, 1);
Xp_64 = (2*pilot_bits_64(1:N)-1) + 1j*(2*pilot_bits_64(N+1:end)-1);
Xp_64 = Xp_64 / sqrt(2);

pilot_idx_8 = 1:8:N;
Xp_8 = zeros(N,1);

pilot_bits_8 = randi([0 1], 2*length(pilot_idx_8), 1);
Xp_temp = (2*pilot_bits_8(1:length(pilot_idx_8))-1) + ...
          1j*(2*pilot_bits_8(length(pilot_idx_8)+1:end)-1);
Xp_temp = Xp_temp / sqrt(2);

Xp_8(pilot_idx_8) = Xp_temp;

[X_64_cp, Y_64_cp] = generate_dataset(Xp_64, [], true,H_bank, delay_bank, cfg, num_train, snr_range);
save('train_64_cp.mat','X_64_cp','Y_64_cp','-v7.3');


[X_64_noCP, Y_64_noCP] = generate_dataset(Xp_64, [], false,H_bank, delay_bank, cfg, num_train, snr_range);
save('train_64_noCP.mat','X_64_noCP','Y_64_noCP','-v7.3');


[X_8_cp, Y_8_cp] = generate_dataset(Xp_8, pilot_idx_8, true,H_bank, delay_bank, cfg, num_train, snr_range);
save('train_8_cp.mat','X_8_cp','Y_8_cp','-v7.3');

disp('Data generated');

%% Helper functions

function [X_out, Y_out] = generate_dataset(Xp, pilot_idx, use_cp,H_bank, delay_bank, cfg, num_train, snr_range)

    N = 64;
    cp_len = 16;

    num_channels = length(H_bank);

    X_out = zeros(num_train, 4*N);
    Y_out = zeros(num_train, 2*N);

    parfor i = 1:num_train

       
        snr_dB = snr_range(1) + rand*(snr_range(2)-snr_range(1));

       
        bits = randi([0 1], 2*N, 1);

        Xd = (2*bits(1:N)-1) + 1j*(2*bits(N+1:end)-1);
        Xd = Xd / sqrt(2);

        if isempty(pilot_idx)
            X = [Xp, Xd];
        else
            Xp_full = zeros(N,1);
            Xp_full(pilot_idx) = Xp(pilot_idx);

            Xd(pilot_idx) = 0; 
            X = [Xp_full + Xd, Xd];
        end

        x_time = ifft(X, N);

        if use_cp
            x_cp = [x_time(end-cp_len+1:end,:); x_time];
        else
            x_cp = x_time;
        end

        x_serial = x_cp(:);

        idx = randi(num_channels);

        H = H_bank{idx};
        delays = delay_bank{idx};

        h_full = squeeze(H{1});
        h_full = h_full(:,1);

        delay_samples = round(delays / cfg.DelaySamplingInterval);

        h = zeros(N,1);

        for k = 1:length(h_full)
            d = delay_samples(k);
            if d <= 16
                h(d+1) = h(d+1) + h_full(k);
            end
        end

        y = conv(x_serial, h);
        y = y(1:length(x_serial));

        signal_power = mean(abs(y).^2);
        noise_var = signal_power / (10^(snr_dB/10));

        noise = sqrt(noise_var/2) * ...
            (randn(size(y)) + 1j*randn(size(y)));

        y = y + noise;

        if use_cp
            y_blocks = reshape(y, N+cp_len, []);
            y_noCP = y_blocks(cp_len+1:end,:);
        else
            y_noCP = reshape(y, N, []);
        end

        Y = fft(y_noCP, N);

        Yp = Y(:,1);
        Yd = Y(:,2);

        X_out(i,:) = [real(Yp); imag(Yp); real(Yd); imag(Yd)].';
        Y_out(i,:) = bits.';

    end
end