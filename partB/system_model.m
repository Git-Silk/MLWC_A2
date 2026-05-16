clear; clc; close all;
rng(42);                    % Fix seed – reproducibility

%  SYSTEM PARAMETERS
M      = 4;                 % Number of BS antennas (ULA)
Nx     = 16;  Ny = 16;      % RIS elements: horizontal x vertical
N      = Nx * Ny;           % Total RIS elements = 256
K      = 1;                 % Number of users (single user)
Nsub   = 64;                % OFDM subcarriers
CP     = 16;                % Cyclic prefix length
L      = 20;                % multipath components
fc     = 28e9;              % Carrier frequency [Hz]  – mmWave
c      = 3e8;               % Speed of light [m/s]
lambda = c / fc;            % Wavelength [m]
d      = lambda / 2;        % Element spacing [m]
bps    = 2;                 % Bits per symbol (QPSK)
L_eff  = min(L, CP);        % Effective CIR taps kept within CP = 16


omega = 2*pi * rand(N, 1);          % Random phase for each RIS element
Phi   = diag(exp(1j * omega));      % Diagonal phase shift matrix (N×N)

H1 = zeros(N, M);
for l = 1:L_eff
    alpha1    = (randn + 1j*randn) / sqrt(2);    % Complex path gain
    phi_AoA   = (rand - 0.5) * pi;              % Azimuth AoA at BS
    phi_AoD   = (rand - 0.5) * pi;              % Azimuth AoD at RIS
    theta_AoD = (rand - 0.5) * pi;              % Zenith  AoD at RIS
    at        = ula_response(M,  phi_AoA,            d, lambda);  % (M×1)
    ar        = upa_response(Nx, Ny, phi_AoD, theta_AoD, d, lambda); % (N×1)
    H1        = H1 + alpha1 * (ar * at');
end
H1 = sqrt(M * N / L_eff) * H1;     % Normalization factor

H2 = zeros(K, N);
for l = 1:L_eff
    alpha2     = (randn + 1j*randn) / sqrt(2);
    phi_AoD2   = (rand - 0.5) * pi;
    theta_AoD2 = (rand - 0.5) * pi;
    ar2        = upa_response(Nx, Ny, phi_AoD2, theta_AoD2, d, lambda); % (N×1)
    H2         = H2 + alpha2 * ar2';    % ar2' is (1×N)
end
H2 = sqrt(N / L_eff) * H2;

Heff = H2 * Phi * H1;           % Effective cascaded channel (1×M)
disp(Heff);

p     = Heff' / norm(Heff);     % (M×1) normalised precoder
g_eff = Heff * p;               % Effective scalar channel (complex)


pwr_dly = exp(-(0:L_eff-1)' / L_eff);   % Exponential PDP
pwr_dly = pwr_dly / sum(pwr_dly);        % Normalize  Σ p_l = 1

h_cir  = (abs(g_eff)) * sqrt(pwr_dly / 2) .* ...
          (randn(L_eff, 1) + 1j*randn(L_eff, 1));

H_freq = fft([h_cir; zeros(Nsub - L_eff, 1)], Nsub);   % (Nsub × 1)

SNR_test_dB = 20;
sigma2      = 10^(-SNR_test_dB / 10);   % Noise variance (signal power = 1)

bits_tx = randi([0 1], Nsub * bps, 1); % Random information bits
sym_tx  = qpsk_mod(bits_tx);           % QPSK symbols         (Nsub × 1)
tx_td   = ifft(sym_tx, Nsub);          % IDFT → time domain
tx_cp   = [tx_td(end-CP+1:end); tx_td];% Insert cyclic prefix (Nsub+CP)

noise_d = sqrt(sigma2/2) * (randn(Nsub,1) + 1j*randn(Nsub,1));
Y_rx    = H_freq .* sym_tx + noise_d;  % Received signal (freq domain)

sym_eq  = Y_rx ./ H_freq;              % ZF equalizer
bits_rx = qpsk_demod(sym_eq);          % Hard-decision demodulation

BER = mean(bits_tx ~= bits_rx);
SER = mean(any(reshape(bits_tx ~= bits_rx, bps, []), 1));

figure('Name','Task 1 – Channel Response','Position',[80 80 950 380]);

subplot(1,2,1);
stem(0:L_eff-1, abs(h_cir), 'filled', 'LineWidth', 1.8, 'Color', [0 0.45 0.74]);
xlabel('Tap Index  n',              'FontSize', 12);
ylabel('|h_{casc}[n]|',            'FontSize', 12);
title('Cascaded Channel Impulse Response (CIR)', 'FontSize', 12);
grid on; xlim([-0.5, L_eff-0.5]);
set(gca,'FontSize',11);

subplot(1,2,2);
plot(0:Nsub-1, abs(H_freq), 'b-', 'LineWidth', 1.8);
xlabel('Subcarrier Index  k',       'FontSize', 12);
ylabel('|H_{eff}[k]|',             'FontSize', 12);
title('Cascaded Channel Frequency Response', 'FontSize', 12);
grid on; xlim([0, Nsub-1]);
set(gca,'FontSize',11);

sgtitle('RIS-Assisted MU-MISO — mmWave Cascaded Channel', ...
        'FontSize', 13, 'FontWeight', 'bold');

% LOCAL FUNCTIONS

function a = ula_response(M, phi, d, lambda)
    idx = (0:M-1)';
    a   = (1/sqrt(M)) * exp(1j * 2*pi * (d/lambda) * idx * sin(phi));
end

function a = upa_response(Nx, Ny, phi, theta, d, lambda)
    a   = zeros(Nx*Ny, 1);
    idx = 1;
    for beta = 0:Ny-1
        for alpha = 0:Nx-1
            a(idx) = exp(1j * 2*pi * (d/lambda) * ...
                (alpha * sin(phi)*sin(theta) + beta * cos(theta)));
            idx = idx + 1;
        end
    end
    a = a / sqrt(Nx * Ny);
end

function sym = qpsk_mod(bits)
    bits = reshape(bits, 2, []);
    I    = 1 - 2*bits(1,:);        % bit=0 → +1,  bit=1 → -1
    Q    = 1 - 2*bits(2,:);
    sym  = (I + 1j*Q).' / sqrt(2); % Normalised power = 1
end

function bits = qpsk_demod(sym)
    sym  = sym(:) * sqrt(2);
    b1   = double(real(sym) < 0);  % I-branch decision
    b2   = double(imag(sym) < 0);  % Q-branch decision
    bits = reshape([b1'; b2'], [], 1);
end
