%'UTF-8'
% Author: Neda Kaboodvand and Behzad Iravani
% n.kaboodvand@gmail.com
% behzadiravani@gmail.com
%------------------------------
classdef Grad_CAM

    properties
        D
        ft_d(1,1) struct % field trip structure
        error(1,1) struct % field trip structure
        hit(1,1) struct % field trip structure

    end

    methods
        function G = Grad_CAM(D)
            % D is CNNdata object
            G.D = D;
        end
        function obj = create_fieldtrip(obj)
            [~, cid] = max(obj.D.data_partitioned_prepratory.y_test_r,[],2);
            obj.ft_d = struct();
            obj.ft_d.label = {'Grad-CAM'};
            obj.ft_d.time  = repmat({obj.D.data_incong_hit_err.data.time(1:size(double(obj.D.grads_data_time.CAMS),2))},1, length(cid));
            obj.ft_d.trial = cellfun(@squeeze, num2cell(double(obj.D.grads_data_time.CAMS),[2,3]), 'UniformOutput',false)';
            obj.ft_d.dimord = 'chan_time';
            obj.ft_d.trialinfo = cid;
            % zscore GradCAMS across the time dimension
            obj.ft_d = obj.zscore_trials(obj.ft_d);

        end

        function obj = divide_to_hit_error(obj)
            % error
            obj.error = FFT_2nd_Analysis.select(obj.ft_d, 1, 'no');
            % hit
            obj.hit = FFT_2nd_Analysis.select(obj.ft_d, 2, 'no');
        end

    end
    methods (Static)
        function o = zscore_trials(ft_d)
            % ZSCORE_TRIALS performs z-scoring on the trial level
            % Input:
            %    ft_d:  fieldtrip strcture
            % -----------------------------
            o = ft_d;
            for tr = 1:numel(ft_d.trial)
                o.trial{tr} = zscore(ft_d.trial{tr}, [],2);
            end % for
        end % function zscore
        function o = hilbert(S, lp, pad, timelock)
            if ischar(lp)
                if strcmpi(lp, 'no')
                    cfg = [];
                    cfg.hilbert = 'abs';
                    cfg.padding = pad;
                    o = ft_preprocessing(cfg, S);

                elseif strcmpi(lp, 'yes')
                    cfg = [];
                    cfg.lpfilter = 'yes';
                    cfg.lpfreq= 12;
                    cfg.hilbert = 'abs';
                    cfg.padding = pad;
                    cfg.demean = 'yes';
                    o = ft_preprocessing(cfg, S);
                end

            else
                cfg = [];
                cfg.lpfilter = 'yes';
                cfg.lpfreq= lp;
                cfg.hilbert = 'abs';
                cfg.padding = pad;
                cfg.demean = 'yes';
                o = ft_preprocessing(cfg, S);
            end

            if strcmpi(timelock, 'yes')
                cfg = [];
                cfg.keeptrials         = 'yes';
                o = ft_timelockanalysis(cfg, o);
            end
        end % end hilbert
        
        % hilbert
        function freq = spect_hibert(S)
        cfg = [];
        cfg.method     = 'hilbert';
        cfg.foi        = 2.^(linspace(2,7.85,30));
        cfg.toi        = S.time{1};
        cfg.keeptrials = 'yes';
        cfg.edgartnan  = 'yes';
        cfg.bpfilttype = 'fir';
        cfg.bpfiltord  = 128*ones(1, length(cfg.foi));
        cfg.pad        = 'nextpow2';

        [freq]         = ft_freqanalysis(cfg,S);
        end

    end % end methods (Static)

end % class
% $END





