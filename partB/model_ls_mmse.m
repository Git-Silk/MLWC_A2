clc; clear; close all;
rng(7);

%% =========================
% System parameters
% =========================
fc      = 28e9;          % carrier frequency
c0      = 3e8;
lambda  = c0 / fc;
d       = lambda / 2;    % antenna spacing

M       = 4;             % BS antennas
Nx      = 16;            % RIS x-dimension
Ny      = 16;            % RIS y-dimension
Nris    = Nx * Ny;       % RIS elements
L       = 20;            % number of mmWave paths

Nsc     = 64;            % OFDM subcarriers
Nfft    = 64;            % FFT size
Ncp     = 16;            % cyclic prefix
Np      = 64;            % pilot subcarriers (assignment says 64)

snrDbVec   = 0:5:25;
numFrames   = 1000;       % increase for smoother curves
pilotIdx    = (1:Np).';  % all subcarriers are pilots

% Transmit beamformer for single-user MISO
wTx = ones(M,1) / sqrt(M);

%% =========================
% Storage
% =========================
berLS   = zeros(size(snrDbVec));
berMMSE = zeros(size(snrDbVec));
mseLS   = zeros(size(snrDbVec));
mseMMSE = zeros(size(snrDbVec));

%% =========================
% Main loop
% =========================
for si = 1:numel(snrDbVec)
    snrDb  = snrDbVec(si);
    sigma2 = 10^(-snrDb/10);   % unit symbol power

    bitErrLS   = 0;
    bitErrMMSE = 0;
    totalBits  = 0;

    mseSumLS   = 0;
    mseSumMMSE = 0;

    for frm = 1:numFrames
        % Generate one effective cascaded RIS channel realization
        [hTime, Hf, Rf] = generate_ris_effective_channel( ...
            M, Nx, Ny, L, Nsc, Nfft, lambda, d, wTx);

        % ==========================================================
        % Pilot OFDM symbol
        % ==========================================================
        XpBits = []; 
        Xp = ones(Nsc,1);  % 64 pilots, one per subcarrier

        txPilot = ofdm_modulate(Xp, Ncp);
        rxPilot = apply_channel(txPilot, hTime, sigma2);

        Yp = ofdm_demodulate(rxPilot, Nfft, Ncp);

        % LS estimate on all subcarriers
        Hls = Yp ./ Xp;

        % LMMSE/MMSE estimate
        % General form: H_hat = R_hp * (R_pp + sigma^2 I)^(-1) * y_p
        % Here, with all-subcarrier pilots and Xp = 1, this simplifies well,
        % but we keep the general form.
        Hmmse = lmmse_estimate(Yp, Xp, pilotIdx, Rf, sigma2, Nsc);

        % ==========================================================
        % Data OFDM symbol
        % ==========================================================
        txBits = randi([0 1], 2*Nsc, 1);
        Xd = qpsk_mod(txBits);

        txData = ofdm_modulate(Xd, Ncp);
        rxData = apply_channel(txData, hTime, sigma2);

        Yd = ofdm_demodulate(rxData, Nfft, Ncp);

        % One-tap equalization
        XhatLS   = Yd ./ Hls;
        XhatMMSE = Yd ./ Hmmse;

        % Hard QPSK decisions
        bitsHatLS   = qpsk_demod(XhatLS);
        bitsHatMMSE = qpsk_demod(XhatMMSE);

        bitErrLS   = bitErrLS   + sum(bitsHatLS   ~= txBits);
        bitErrMMSE = bitErrMMSE + sum(bitsHatMMSE ~= txBits);
        totalBits   = totalBits   + numel(txBits);

        % Channel estimation MSE
        mseSumLS   = mseSumLS   + mean(abs(Hls   - Hf).^2);
        mseSumMMSE = mseSumMMSE + mean(abs(Hmmse - Hf).^2);
    end

    berLS(si)   = bitErrLS / totalBits;
    berMMSE(si) = bitErrMMSE / totalBits;

    mseLS(si)   = mseSumLS / numFrames;
    mseMMSE(si) = mseSumMMSE / numFrames;

    fprintf('SNR = %2d dB | BER-LS = %.4e | BER-MMSE = %.4e\n', ...
        snrDb, berLS(si), berMMSE(si));
end

%% =========================
% Plots
% =========================
figure;
semilogy(snrDbVec, berLS, '-o', 'LineWidth', 1.6); hold on;
semilogy(snrDbVec, berMMSE, '-s', 'LineWidth', 1.6);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('RIS-assisted MU-MISO OFDM: BER vs SNR');
legend('LS', 'MMSE', 'Location', 'southwest');

figure;
semilogy(snrDbVec, mseLS, '-o', 'LineWidth', 1.6); hold on;
semilogy(snrDbVec, mseMMSE, '-s', 'LineWidth', 1.6);
grid on;
xlabel('SNR (dB)');
ylabel('Channel MSE');
title('RIS-assisted MU-MISO OFDM: Channel Estimation MSE vs SNR');
legend('LS', 'MMSE', 'Location', 'southwest');

%% =========================================================
% Local functions
% =========================================================

function tx = ofdm_modulate(Xf, Ncp)
    % Xf: Nsc x 1 frequency-domain vector
    xt = ifft(Xf, [], 1);
    tx = [xt(end-Ncp+1:end); xt];
end

function Yf = ofdm_demodulate(rx, Nfft, Ncp)
    % Remove CP and FFT
    rxNoCP = rx(Ncp+1:Ncp+Nfft);
    Yf = fft(rxNoCP, Nfft);
end

function y = apply_channel(x, h, sigma2)
    % Linear convolution + AWGN
    yClean = conv(x, h);
    yClean = yClean(1:numel(x)); % truncate to symbol length
    noise = sqrt(sigma2/2) * (randn(size(yClean)) + 1j*randn(size(yClean)));
    y = yClean + noise;
end

function [hTime, Hf, Rf] = generate_ris_effective_channel( ...
    M, Nx, Ny, L, Nsc, Nfft, lambda, d, wTx)

    Nris = Nx * Ny;

    % Random RIS phase shifts as required in the assignment
    phiRIS = exp(1j * 2*pi * rand(Nris,1));

    % Discrete delays fit within CP=16
    % Use 0...15 so CP=16 covers the spread.
    delays = sort(randi([0, 15], L, 1));

    % Path gains
    alpha1 = (randn(L,1) + 1j*randn(L,1)) / sqrt(2);
    alpha2 = (randn(L,1) + 1j*randn(L,1)) / sqrt(2);

    % Make first path slightly stronger
    alpha1(1) = 2 * alpha1(1);

    % Effective tap gains
    tapGain = zeros(L,1);

    for l = 1:L
        % Random AoA/AoD angles
        azBS   = 2*pi*rand;
        azR1   = 2*pi*rand;
        zeR1   = pi*rand;
        azR2   = 2*pi*rand;
        zeR2   = pi*rand;

        aBS  = ula_sv(M, azBS, lambda, d);
        aR1  = upa_sv(Nx, Ny, azR1, zeR1, lambda, d);
        aR2  = upa_sv(Nx, Ny, azR2, zeR2, lambda, d);

        % BS -> RIS path contribution
        H1_l = sqrt(M * Nris / L) * alpha1(l) * (aR1 * aBS.');

        % RIS -> UE path contribution
        h2_l = sqrt(Nris / L) * alpha2(l) * aR2;

        % Cascaded effective scalar path gain after beamforming
        tapGain(l) = h2_l' * (phiRIS .* (H1_l * wTx));
    end

    % Build discrete-time channel impulse response
    hTime = zeros(Nfft,1);
    for l = 1:L
        idx = delays(l) + 1;     % MATLAB indexing
        hTime(idx) = hTime(idx) + tapGain(l);
    end

    % Normalize average power
    hTime = hTime / sqrt(sum(abs(hTime).^2) + eps);

    % Frequency response
    Hf = fft(hTime, Nsc);

    % Build frequency-domain covariance for LMMSE
    % Use tap-power spectrum from the discrete taps
    pTap = abs(hTime).^2;
    pTap = pTap / (sum(pTap) + eps);

    Rf = zeros(Nsc, Nsc);
    for m = 1:Nsc
        for n = 1:Nsc
            acc = 0;
            for q = 1:Nfft
                if pTap(q) > 0
                    tau = q - 1;
                    acc = acc + pTap(q) * exp(-1j * 2*pi * (m-n) * tau / Nfft);
                end
            end
            Rf(m,n) = acc;
        end
    end

    Rf = (Rf + Rf') / 2;
    Rf = Rf / (real(trace(Rf))/Nsc + eps);
end

function Hhat = lmmse_estimate(Yp, Xp, pilotIdx, Rf, sigma2, Nsc)
    % General LMMSE/MMSE estimate from pilot observations.
    % Yp = diag(Xp) * H + n
    %
    % For full pilots (pilotIdx = 1:Nsc), this reduces to a simple form.
    %
    % Hhat = R_hp * inv(R_pp + sigma2*I) * y_p
    %
    % Since the pilot symbols are known, we equalize the pilot observations
    % first to obtain an LS estimate, then use covariance smoothing.

    Hls_p = Yp ./ Xp;

    Rpp = Rf(pilotIdx, pilotIdx);
    Rhp = Rf(:, pilotIdx);

    Hhat = Rhp * ((Rpp + sigma2 * eye(numel(pilotIdx))) \ Hls_p);

    % Normalize if numerical scaling drifts
    Hhat = Hhat / sqrt(mean(abs(Hhat).^2) + eps);
end

function a = ula_sv(M, az, lambda, d)
    n = (0:M-1).';
    a = exp(1j * 2*pi * (d/lambda) * n * sin(az)) / sqrt(M);
end

function a = upa_sv(Nx, Ny, az, ze, lambda, d)
    [ix, iy] = meshgrid(0:Nx-1, 0:Ny-1);
    ix = ix(:);
    iy = iy(:);

    phase = 2*pi * (d/lambda) * (ix .* sin(az).*sin(ze) + iy .* cos(ze));
    a = exp(1j * phase) / sqrt(Nx * Ny);
end

function s = qpsk_mod(bits)
    bits = bits(:);
    if mod(numel(bits),2) ~= 0
        bits(end+1) = 0;
    end
    b = reshape(bits, 2, []).';
    s = ((1 - 2*b(:,1)) + 1j*(1 - 2*b(:,2))) / sqrt(2);
end

function bits = qpsk_demod(sym)
    b1 = real(sym) < 0;
    b2 = imag(sym) < 0;
    bits = reshape([b1.'; b2.'], [], 1);
end