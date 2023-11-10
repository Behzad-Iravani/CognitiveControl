% -*- code: 'UTF-8' -*-
classdef results_stimulation < handle
    % results_stimulation is class for ploting the stimulation results
    % This script is part of the analysis for conflic error processing and SCNN
    % Author: Neda Kaboodvand and Behzad Iravani
    % n.kaboodvand@gmail.com
    % behzadiravani@gmail.com
    properties
        path_stim(1,2) cell                         % - a cell contains a path to the parameters range and stimulation simulations

    end % end properties

    properties(Dependent)
        args                                         % - a matrix contains double values determining the range of parameter search
        search                                       % - a matrix contains dobule values of prediction for the search range
        TCs                                          % - a matirx contains the simulated time-series for the optimized C
        MAE                                          % - a vector contains the order of nodes based on the mean errors
        Op                                           % - a structure contains the maximum value of hit score and associated index of C
    end % dependent properties

    methods
        function obj = results_stimulation(path_stim)
            obj.path_stim = path_stim;
        end % constructor

        function plot_parameters(obj)
            xx = interp1(1:200,obj.args(:,1),linspace(1,200,1e3),'nearest');
            yy = smooth(interp1(1:200,obj.search(:,end),linspace(1,200,1e3),'nearest'),30,'sgolay');
            plot(xx... interpolating the search range using pchip method
                ,yy, Color='k') % interploating the hit score (:, end) using pchip method
            hold on
            scatter(xx(1:9:end), yy(1:9:end), Marker="square", MarkerEdgeColor='k',  MarkerFaceColor='k')
            hold off
            xlim([.00,.03])
            line([xlim()], [.5 .5], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.25)
            ylabel('Hit Accuracy')
            title('\rmDecoding simulated signals by SCNN')
            ax = gca();
            obj.plot_settings(ax,...
                [min(obj.args(:,1)):.005:max(obj.args(:,1))],...
                [min(obj.args(:,1)):.001:max(obj.args(:,1))],...
                [0:.3:.6], [0:.05:1], 18)
            print -dsvg -vector prediction_dm_by_SCNN.svg
        end % plot_paramters

        function plot_stimulatation_inslico_TCs(obj)
            clf
            col = nodescol();
            hold on
            % original
            for i = 1:size(obj.MAE,2)
                plot(linspace(-1.5,0,size(obj.TCs,2)),...
                    squeeze(obj.TCs(:,:,obj.MAE(i))) + repmat(i, size(obj.TCs,2),1)', 'Color', col(obj.MAE(i),:), 'LineWidth', 1.25)
            end

            hold off
            xlim([-1.5, 0])
            xlabel('Time(s)')
            title('\rmSimulated time-series')
            ax = gca();
            obj.plot_settings(ax, ...
                [-1.5:.5:0.1], [-1.5:.1:0.1],[1:19], [1:19], 16)
            ax.YTickLabel = 19:-1:1;

            print -dsvg -vector simulated_TC.svg
        end % plot_stimulatation_inslico_TCs
        function report  = report_stat(obj, nperm)
            rng default % for reproducibility
            iperm = randi(length(obj.search(:,end)), nperm, 1);
            mean(obj.search(iperm ,end))
            report.tval = (obj.search(:,end)- mean(obj.search(iperm ,end)))./std(obj.search(iperm ,end));
            report.pval = mean(obj.search(:,end)<=obj.search(iperm ,end)',2);
            [~, imx]= max(report.tval);
            fprintf('Hieghest acc at c: %1.4f : t = %1.2f, p = %1.3f\n',obj.args(imx ,1), report.tval(imx),  report.pval(imx))
        end
        %%---------------------------------------------------------------%%
        %%---------------------------------------------------------------%%
        % get methods
        function search = get.search(obj)
            ls = dir(strcat(obj.path_stim{1,1},'search_*.mat'));
            search =[];
            for sn = 1:numel(ls)
                load(fullfile(ls(sn).folder, sprintf('search_%d', sn-1)));
                search = [search; output];
            end
        end % get search
        %%---------------------------------------------------------------%%
        function args = get.args(obj)
            load(strcat(obj.path_stim{1,1},'_args_C.mat'))
            %         args = args;
        end % get args
        %%---------------------------------------------------------------%%
        function TCs = get.TCs(obj)

            ls = dir(strcat(obj.path_stim{1}, 'simulated_', num2str(obj.Op.ix-1), '.mat')); % convert indexing to python format (obj.MAE-1)
            TCs = [];
            for sn = 1:numel(ls)
                load(fullfile(ls(sn).folder, ls(sn).name));
                TCs = cat(1,TCs, dat_downsample);
            end
        end % get TCs
        %%---------------------------------------------------------------%%
        function MAE = get.MAE(obj)
            load(fullfile(obj.path_stim{1,1}, 'orderMAE.mat'))
            MAE = ix;
        end % get MAE
        function Op = get.Op(obj)
            [Op.max, Op.ix] = max(obj.search(:,end));
        end % get Op
        function stimulating_three_first_rois(obj, regions) % ["ROI 12-1", "ROI 2-2", "ROI 18-3", "ROI 16-4"]
            clc
            iplot = 0;
            n = ceil(sqrt(length(regions)));
            for region = regions
                iplot = iplot +1;
                search = [];
                for amp = linspace(.5,5, 20)
                    tmpsearch = [];
                    for freq = linspace(2,200, 20)

                        load(strcat(string(obj.path_stim{2}), "\Stimulation_", region ,"\",sprintf("_Stimulation_search_freq%1.1f_%1.1f.mat", 1e3*freq, amp)))
                        tmpsearch = [tmpsearch; output];
                    end
                    search  = cat(3, search, tmpsearch);
                end
                hold on
                col_ = plasma(32);%col(12,:);
                clear search_
                for fi = 1:20
                    search_(fi,:) =  smooth(squeeze(search(fi,end,:)),3, 'moving' );
                end

                % ploting
                subplot(n,n,iplot)
                imagesc(linspace(.5,5, 20),...
                    linspace(2,200, 20),...
                    (search_))

                % axis xy
                ax(iplot) = gca();
                obj.plot_settings(ax(iplot), linspace(.5,5, 4), linspace(.5,2.5, 16), [2,50:50:200], [25:25:200], 16);
                ax(iplot).XAxis.TickLabels= cellfun(@(x)sprintf("%1.2f",x),num2cell(linspace(.5,5, 4)));


                clim([.53, .63])
                cb = colormap(col_);
                axis xy tight
                if iplot == 1
                    xlabel('Normalized amplitude', 'FontSize', 14);
                    ylabel('Frequency (Hz)', 'FontSize', 14);
                    cb = colorbar();
                    cb.Ticks = [.53:.1:.65];

                else
                    set(ax(iplot),'XTickLabel', [], 'YTickLabel', []);
                    ax(iplot).Position(3:4) =  ax(1).Position(3:4);

                end
                drawnow();
                title(regions(iplot));
            end % for regions

        end % stimulating_three_first_rois
        function [t, p , cirange] = stimulating_rois_stat(obj, regions) % ["ROI 12-1", "ROI 2-2", "ROI 18-3", "ROI 16-4"]
            clc
            roi = 0;
            for region = regions
                roi= roi +1;
                search = [];
                for amp = linspace(.5,5, 20)
                    tmpsearch = [];
                    for freq = linspace(2,200, 20)

                        load(strcat(string(obj.path_stim{2}), "\Stimulation_", region ,"\",sprintf("_Stimulation_search_freq%1.1f_%1.1f.mat", 1e3*freq, amp)))
                        tmpsearch = [tmpsearch; output];
                    end
                    search  = cat(3, search, tmpsearch);
                end
                clear search_
                for fi = 1:20
                    search_(fi,:) =  smooth(squeeze(search(fi,end,:)),3, 'moving' );
                end
                dat(:, :, roi) = search_;
            end % region

            [t, p , cirange] = obj.permutation(dat, 5e3,3);
        end % stimulating_rois_stat

        function s = stimulating_traceplot(obj, regions)
            iregion = 0;
            for region = regions
                iregion = iregion + 1;
                search = [];
                for amp = linspace(.5,5, 20)
                    tmpsearch = [];
                    for freq = linspace(2,200, 20)

                        load(strcat(string(obj.path_stim{2}), "\Stimulation_", region ,"\",sprintf("_Stimulation_search_freq%1.1f_%1.1f.mat", 1e3*freq, amp)))
                        tmpsearch = [tmpsearch; output];
                    end
                    search  = cat(3, search, tmpsearch);
                end

                col_ = plasma(32);%col(12,:);
                clear search_
                for fi = 1:20
                    search_(fi,:) =  smooth(squeeze(search(fi,end,:)),3, 'moving' );
                end
                dat(iregion) = max(search_, [], "all");
            end % regions
            % ploting
            col = nodescol();
            hold on
            iplot = 0;
            s.mu = mean(dat);
            s.sigma = std(dat);
            s.df = length(dat)-1;
            % Calculate the t-transformed values using a degrees of freedom of 19
            s.t = (dat - s.mu) ./ (s.sigma / sqrt(length(dat)));
            s.p = 2*(1-tcdf(abs(s.t), s.df));
            t_critical = tinv(1-.025, s.df);
            s.ci =[s.mu-  t_critical*(s.sigma / sqrt(length(dat))), s.mu + t_critical*(s.sigma / sqrt(length(dat)))];

            for iplot = 1:length(dat)
                bar(iplot, s.t(iplot), 'FaceColor', col(obj.MAE(end-iplot+1),:))
            end
            obj.plot_settings(gca,  1:19, [1:19],-6:3:12,  [-6:1:12], 16)
            ylim([-7,12])
            pbaspect([1,.45,1])
            hold off
        end %  stimulating_traceplot

        function TCstimulate = get_TCstimulate(obj, region_, freq, amp)

            load(strcat("source\Stimulation_", region_ ,"\",sprintf("_Stimulation_search_freq%1.1f_%1.1f.mat", freq, amp)))
            TCstimulate = dat_downsample;

            col = nodescol();
            hold on
            % original
            for i = 1:size(obj.MAE,2)
                col_tmp = obj.darken(col(obj.MAE(i),:));
                plot(linspace(-1.5,0,size(TCstimulate,2)),...
                    squeeze(TCstimulate(:,:,obj.MAE(i))) + repmat(i, size(TCstimulate,2),1)', 'Color', col_tmp ,...
                    'LineWidth', 1.5, 'LineStyle', '--')
            end

            hold off
            xlim([-1.5, 0])
            xlabel('Time(s)')
            title(['\rmSimulated time-series: ',sprintf('f = %1.1f, a = %1.1f', freq/1e3, amp)])
            ax = gca();
            obj.plot_settings(ax, ...
                [-1.5:.5:0.1], [-1.5:.1:0.1],[1:19], [1:19], 18)

        end % get TCstimualte
          function ax =implot(obj, t, lim) % ["ROI 12-1", "ROI 2-2", "ROI 18-3", "ROI 16-4"]
            clc
            col_ = plasma(32);%col(12,:);

            imagesc(linspace(.5,5, 20),...
                linspace(2,200, 20),...
                (t))

            % axis xy
            ax = gca();
            obj.plot_settings(ax, linspace(.5,5, 4), linspace(.5, 5, 16),[2,50:50:200], [25:25:200], 16);
            ax.XAxis.TickLabels= cellfun(@(x)sprintf("%1.2f",x),num2cell(linspace(.5,5, 4)));


            clim(lim)
            cb = colormap(col_);
            axis xy tight

            xlabel('Normalized amplitude', 'FontSize', 14);
            ylabel('Frequency (Hz)', 'FontSize', 14);
            cb = colorbar();
            cb.Ticks = [linspace(lim(1),lim(2),3)];

            drawnow();
            title('Statistical map');
        end % implot

    end %  methods

    methods(Access = private)

        function dc = darken(~, c)
            c = rgb2hsv(c);
            c(1) = 1-c(1);
            c(3) = .98*c(3);
            dc = hsv2rgb(c);
        end
        function ax = plot_settings(~, ax, xtick, xmtick, ytick, ymtick, fontsize)
            % plot_settings is a private method of class result_stimulation
            % Inputs:
            %           ax      - axis handel
            %           xtick   - a slice determinig the major xticks
            %           xmtick  - a slice determinig the minor xticks
            %           ytick   - a slice determinig the major yticks
            %           ymtick  - a slice determinig the minor yticks
            %           fontsize- an integer defines the fontsize
            arguments
                ~
                ax
                xtick
                xmtick
                ytick
                ymtick
                fontsize(1,1) double {mustBeInteger(fontsize)}
            end

            ax.FontSize =  fontsize;
            ax.FontName = 'Arial';
            ax.LineWidth= 2.25;
            ax.TickLength(1) = 0.04;

            ax.YAxis.TickValues= ytick;
            ax.YAxis.MinorTick = 'on';
            ax.YAxis.MinorTickValues = ymtick;%.60:.1:1;
            ax.XAxis.TickValues= xtick;
            ax.XAxis.MinorTick = 'on';
            ax.XAxis.MinorTickValues = xmtick;
            box off
            axis square
        end

    end % private
    methods(Static)
        function [t, p , cirange] = permutation(dat, nrep, dim)
            rng(1)
            design = randi(size(dat, dim), nrep,1);
            for i = 1:size(dat, dim)
                t(:,:,i) = (dat(:,:, i) - mean(dat(:,:, design), 3))./std(dat(:,:, design), [], 3);
                p(:,:,i) = mean(repmat(dat(:,:, i), 1, 1, nrep) < dat(:,:, design),dim);
                cirange(:,:,i) = 2*std(dat(:,:, design), [], 3)/sqrt(nrep);
            end
        end % permutation

      
    end % methods Static

end  % results_stimulation