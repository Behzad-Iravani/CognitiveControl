classdef FFT_2nd_Analysis
    % This class performs the frequency analysis of the TFR signal
    % dervided from the amplitude signals of the DCNN components

    properties
        grad % gradcam object
        TFRorig % fieldtrip structure contains the the TFR of the amplitude signal
        % based on the learned filters from DCNN
        error
        hit
    end
    properties(Dependent)
        TFRhilbert % fieldtrip structure contains the the TFR of the amplitude signal
        % based on the learned filters from DCNN
    end

    methods
        function FFT2 = FFT_2nd_Analysis(GC, ft_d)
            FFT2.grad    = GC;
            FFT2.TFRorig = ft_d;
        end

        function value = get.TFRhilbert(obj)
            value = Grad_CAM.hilbert(obj.TFRorig,'no',2, 'no');
        end

        function obj = divide_tfr_to_hit_error(obj, Latency)
            % error
            obj.error = obj.select(obj.TFRhilbert, 1, Latency);
            % hit
            obj.hit = obj.select(obj.TFRhilbert, 2, Latency);
        end
      
    end
    methods(Static)
        function o = interp_(ft, npoints, parameters)
            o = ft;
            o.freq = linspace(ft.freq(1), ft.freq(end), npoints);
            for i = 1:numel(parameters)
                if strcmp(parameters{i},'mask')
                    o.(parameters{i}) = logical(interp1(ft.freq,double(ft.(parameters{i})'),o.freq, 'nearest'))';
                else
                    o.(parameters{i}) = interp1(ft.freq,ft.(parameters{i})',o.freq, 'pchip')';
                end % end if
            end % end for
        end % end interp_

        function o = select(ft, label, Latency)
            if ischar(Latency)
                if strcmpi(Latency,'no')
                    cfg = [];
                    cfg.trials = find(ft.trialinfo == label); % error coded by 1 and hit by 2
                    o = ft_selectdata(cfg, ft);
                elseif strcmpi(Latency,'yes')
                    cfg = [];
                    cfg.latency=[-inf, inf];
                    cfg.trials = find(ft.trialinfo == label);
                    o = ft_selectdata(cfg, ft);
                else
                    error('not a valid input for latency!')
                end % if no/yes
            else
                cfg = [];
                cfg.trials = find(ft.trialinfo == label);
                cfg.latency=Latency;
                o = ft_selectdata(cfg, ft);

            end % if ischar
        end % function select
          function [TFRe, TFRh] = run_fft(error, hit)

            %%------TFR ERROR and hit ------
            cfg = [];
            cfg.output       = 'pow';
            cfg.method       = 'mtmfft';
            cfg.keeptrials   = 'yes';
            cfg.taper        = 'dpss';
            cfg.foi          = 0:5:200;                         % analysis 1 to 60 Hz in steps of 5 Hz
            % cfg.t_ftimwin    = 3./cfg.foi;   % length of time window = 0.5 sec
            % cfg.width        = 7./cfg.foi;
            cfg.tapsmofrq    =   10;
            cfg.toi          = 'all';
            cfg.pad          ='nextpow2';
            TFRe = ft_freqanalysis(cfg, error);
            TFRh = ft_freqanalysis(cfg, hit);
        end

    end

end