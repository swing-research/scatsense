%% Load impulse response data from SOFA file into a matrix

SOFAfile = 'data/QU_KEMAR_anechoic_3m.sofa';

obj  = SOFAload(SOFAfile);
data = obj.Data;
Fs = data.SamplingRate;

D = 360; %number of directions
T = 2048; %number of samples in impulse response
M = 2; %number of microphones
h_theta_t = zeros(T,D,M); %matrix of impulse responses

dr = 181;
for d=1:180
    % angles from 0 to 179
    h_theta_t(:,d,1) = squeeze(data.IR(dr,1,:)); %right
    h_theta_t(:,d,2) = squeeze(data.IR(dr,2,:)); %left
    %angles from 180 to 359
    h_theta_t(:,dr,1) = squeeze(data.IR(d,1,:)); %right
    h_theta_t(:,dr,2) = squeeze(data.IR(d,2,:)); %left
   
    dr = dr+ 1;
end

save('kemar_44100_h_theta_t.mat','h_theta_t')
