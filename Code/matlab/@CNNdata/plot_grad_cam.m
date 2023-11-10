% -*- code: 'UTF-8' -*-
% This script is part of the analysis for conflic error processing and SCNN
% plots the GradCAM activity
% Author: Neda Kaboodvand and Behzad Iravani
% n.kaboodvand@gmail.com
% behzadiravani@gmail.com
%%-------------------------------------%%

function s2r = plot_grad_cam(Data, path, fpath, load_)
% Input:
%       Data         CNN data instance 
%       path         path to result folder
%       fpath:       path to figure folders
%       load:        load already computed;
% --------------------------------------
%% Create the GradCAM object
if ~ load_
GC = Grad_CAM(Data);
% create field trip stracture from GradCAM
GC = GC.create_fieldtrip();
% divid to hit and error
GC = GC.divide_to_hit_error(); % devide to error and hit
% % save(fullfile(path, 'GradCAMob.mat'), 'GC', '-v7.3')
%% hilbert transform to extract instantanous amplitude
H.error = GC.spect_hibert(GC.error);%hilbert(GC.error,12,2, 'yes');
H.hit   = GC.spect_hibert(GC.hit);%hilbert(GC.hit,12,2, 'yes');
%% Perfome monte carlo stats for GradCAM
cfg = [];
cfg.method           = 'montecarlo';%'analytic';
cfg.numrandomization = 1e3;


% % cfg.correctm         = 'cluster';
% % cfg.clusteralpha     = 0.05;       % alpha level of the sample-specific test statistic that
% %                                    % will be used for thresholding
% % cfg.clusterstatistic = 'maxsum';

% cfg.resampling       = 'bootstrap';
cfg.statistic        = 'ft_statfun_indepsamplesT';
cfg.alpha            = 0.05;
n1                   = size(H.error.trialinfo,1);
n2                   = size(H.hit.trialinfo,1);
design               = zeros(2,n1+n2);
design(1,1:n1)       = 1;
design(1,n1+1:n1+n2) = 2;
design(2,1:n1)       = [1:n1];
design(2,n1+1:n1+n2) = [1:n2];

cfg.design           = design;
cfg.ivar             = 1;
% cfg.feedback         = 'gui';
% % % cfg.uvar     = 2;
stat                 = ft_freqstatistics(cfg, H.hit, H.error);
save(fullfile(path, 'Grad-CAM_stats.mat'), 'stat','H')

else
 load(fullfile(path, 'Grad-CAM_stats.mat'))
end
%% Plot GradCAM stats
%%---------- PANEL A Figure ---------%%
% cfg = [];
% cfg.parameter = 'stat';
% cfg.maskparameter = 'mask';
% % cfg.linewidth = 2;
% % cfg.linecolor = 'k';
% cfg.maskstyle      = 'opacity';
% ft_singleplotTFR(cfg, stat)
im = imagesc(stat.time, 1:30, squeeze(stat.stat),[0,2.25]);
axis xy
im.AlphaData = squeeze(stat.mask&  stat.stat>0);

Comps = spm_bwlabel(double(squeeze(stat.mask &  stat.stat>0)),6);
ci = 0 ;
for icomp = 1 : max(Comps(:))
        if sum(Comps == icomp, "all")>50
            ci = ci +1;
            clear tmp inds ind ci_;
            inds  = find(Comps == icomp);
            tmp = squeeze(stat.stat);
% % %             ref = squeeze(stat.ref);
% % %             [Hit, Error] = compute_inference(Comps == icomp);
            
            [s2r.t(ci), s2r.ind(ci)] = max(tmp(Comps == icomp),[],'all');
            s2r.p(ci)                = stat.prob(inds(s2r.ind(ci)));

            % Compute the confidence interval range
            cirange = abs(squeeze(stat.cirange));  % Take the absolute value of cirange since it represents a radius
           
            s2r.confidenceIntervalLower(ci) = s2r.p(ci) - cirange(inds(s2r.ind(ci)));  % Multiply the margin of error by 2 to get the confidence interval range
            s2r.confidenceIntervalUpper(ci) = s2r.p(ci) + cirange(inds(s2r.ind(ci))); 

            [ind.f, ind.t]           = ind2sub(size(tmp), inds);
            s2r.f(ci,:)              = [min(stat.freq(ind.f)), max(stat.freq(ind.f))];
        end
end

% plot(stat.time, stat.stat, 'k', 'LineWidth', 1.5)
pbaspect([1, .7, 1])
box off
ax= gca();
ax.FontSize =  20;
ax.FontName = 'Arial Nova Cond';
ax.LineWidth= 1.25;
ax.TickLength(1) = 0.02;

ytc = stat.freq;
ax.YAxis.TickValues= 1:2:30;
ax.YAxis.TickLabels= round([stat.freq(1:2:end)]);

% ax.YAxis.MinorTick = 'on';
% ax.YAxis.MinorTickValues = 2.^(2:.25:7.80);%.60:.1:1;
ax.XAxis.TickValues= [-1.5:.5:0];
ax.XAxis.MinorTick = 'on';
ax.XAxis.MinorTickValues = [-1.5:.1:0];
ylim([1 30])
xlabel('Time (s)')
ylabel('Frequency (Hz)')
cmap = color_();
colormap(cmap(end/2:end,:))
ylabel('T-value')
xlabel('Time(s)')
ax.CLim = [0,2.25];
cbar = colorbar();
ylabel(cbar, 'T-value')
title('Grad-CAM activity (Hit>Error)')
grid on
print(fullfile(fpath,'DS01_grad-cam.svg'),'-dsvg', '-vector')


function cmap = color_()
rgbData = [
  0, 128, 0;    % Green
  16, 112, 0;
  128, 255, 0;  % Yellow
  144, 240, 0;
  255, 255, 255;
  255, 255, 255;
  240, 144, 0
  112, 16, 0;
  128, 0, 0;    % Red
  144, 16, 0;
  200, 5, 0;
 ];
cmap = interp1(1:length(rgbData), rgbData, linspace(1,length(rgbData),64))./255;
end
end