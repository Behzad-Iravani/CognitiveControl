%'UTF-8'
classdef CNNdata < handle
    % CNNdata is class contains the data  for the
    % interpretability of the DCNN anf cognitive control


    % Author: Neda Kaboodvadn and Behzad Iravani
    % n.kaboodvand@gmail.com
    % behzadiravani@gmail.com

    % Copyright (C) Neda Kaboodvand and Behzad Iravani, Departement of
    % neruology and neurological sciences, Stanford USA

    % december 2022, Palo Alto, USA

    properties
        path(1,1) string                    % - string contains the path to metrics (accuracy and loss)
        parameters cell                     % - a cell contains the file names storing data
        accuracy(1,3) cell                  % - a cell contains the final accuracies
        MeanAverageError(1,1) struct        % - a structure contains the mean average error
        MeanAverageError_chan(1,1) struct   % - a structure contains the mean average error per channel
        grads_data_time(1,1) struct         % - a structure contains the gradient activation mapping data
        Kernel_layer
        Maximized_class
        history
        activation
        data_incong_hit_err
        data_partitioned_prepratory
        nodeLabel
        filters

    end

    methods
        function o = CNNdata(path, parameters)
            o.path = path;
            o.parameters = parameters;

        end
        function LoadParameters(obj) % load data
            if nargin > 0
                for i = 1:length(obj.parameters)
                    f = strsplit(obj.parameters{i}, '.');
                    switch f{2}
                        case 'mat'
                            try
                                obj.(f{1}) = load(fullfile(obj.path, obj.parameters{i}));
                                disp([f{1},' is succeffuly loaded!'])
                            catch
                                disp([f{1},'COULD NOT be succeffuly loaded!'])
                            end
                        case {'csv', 'xlsx'}
                            try
                                obj.(f{1}) = readtable(fullfile(obj.path, obj.parameters(i)));
                                disp([f{1},' is succeffuly loaded!'])
                            catch
                                disp([f{1},'COULD NOT be succeffuly loaded!'])
                            end

                        case 'txt'
                            try
                                fid = fileread(fullfile(obj.path, obj.parameters{i}));
                                obj.(f{1}) = regexp(fid, '(?<=(acc of ))\d+.\d+', 'match');
                                disp([f{1},' is succeffuly loaded!'])
                            catch
                                disp([f{1},'COULD NOT be succeffuly loaded!'])
                            end

                    end
                end % end for paramters loop
            end % end if
        end % end function

        %%----------- DIPLAYING ACC -----------%%
        function acc = whats_acc(obj)
            l = {'training', 'cross-val', 'testing'};
            for i = 1:length(obj.accuracy)
                fprintf('%s : %1.1f\n', l{i}, str2num(obj.accuracy{i}))
            end
        end % end function whats_acc
        %%----- Plotting learning and loss ----%%
        function plot_learning_loss(obj)

            figure
            plot(obj.history.Var1, smooth(obj.history.acc,5,'sgolay'), 'LineWidth', 2)
            hold on
            plot(obj.history.Var1, smooth(obj.history.val_acc,5,'sgolay'), 'LineWidth', 2)
            xlabel('Epochs')
            ylabel('Accuracy')
            title('\rm Network accuracy')
            ax = gca();
            xlim([0,50])
            obj.plot_setting( ax, [.40, .70], 16,...
                'Arial', 1.25, .40:.1:.70, 0:25:50, .45:.025:.70, 0:5:50 );% [.45, .75], 16gca()
            legend({'Training', 'Cross validation'}, 'Box', 'off')
            print -dsvg figures/DS01/DS01_accuracy.svg
            %%
            figure
            plot(obj.history.Var1, smooth(obj.history.loss,5,'sgolay'), 'LineWidth', 2)
            hold on
            plot(obj.history.Var1, smooth(obj.history.val_loss,5,'sgolay'), 'LineWidth', 2)
            ax = gca();
            xlim([0,50])
            obj.plot_setting( ax, [.65, 0.85], 16,...
                'Arial', 1.25, .65:.05:.85, 0:25:50, .65:.01:.85, 0:5:50 )
            xlabel('Epochs')
            ylabel('Performance loss')
            title('\rm Network performance loss')


            legend({'Training', 'Cross validation'}, 'Box', 'off')
            print -dsvg figures/DS01/DS01_performance_loss.svg

        end% end function plot_learning_loss
        function  peaks_ = plot_learned_filters(obj, fontsize, ix)

            col = obj.getColor();
            fs = 500;
            colors = eye(3);
            figure('Units','normalized',OuterPosition=[0 0 1 1])
            iplot = 0;
            for i = ix
                iplot = iplot +1;
                subplot(4,5, iplot)
                hold on
                n = length(obj.Kernel_layer.Kernel(:,i));

                f = (-n/2:n/2-1)*fs/n;

                FK = fftshift(fft(obj.Kernel_layer.Kernel(:,i)));
                %                 FK_l = abs(FK).*exp(1i*(linspace(pi,-pi,32)))';
                pow = 20*log10(abs(FK)/n) + 30; % dB to dBM + 30
                %                 pow_l = 20*log10(abs(FK_l)/n) + 30;
                FQ = interp1(1:length(pow),f,linspace(1,length(pow),250));
                AMP = interp1(1:length(pow),smooth(pow, 5, 'sgolay'),...
                    linspace(1,length(pow),250), 'pchip');
                area(FQ, AMP, 'FaceColor', col(i,:), 'EdgeColor', 'w', BaseValue= -40)
                [~, locs, ~,P] = findpeaks(AMP, 'MinPeakHeight', -15);

                peaks_.fq{i} = FQ(locs);
                peaks_.a{i}  = AMP(locs);
                
%                 for ff = 1:length(s2r.f)
%                     if  any(FQ(locs)>= s2r.f(ff,1) & FQ(locs)<= s2r.f(ff,2))
%                         wind = find(FQ(locs)>= s2r.f(ff,:) && FQ(locs)<= s2r.f(ff,:));
%                         scatter(FQ(locs(wind)), AMP(locs(wind))+5, 'filled', 'Marker', 'v', 'MarkerEdgeColor','none','MarkerFaceColor',colors(ff,:));
%                     end
%                 end
                % % %                 figure
                % % %                 area(interp1(1:length(pow),f,linspace(1,length(pow),250)), interp1(1:length(pow),smooth(pow_l, 5, 'sgolay'),...
                % % %                     linspace(1,length(pow),250), 'pchip'), 'FaceColor', col(i,:), 'EdgeColor', 'w', BaseValue= -40)
                ax = gca();
                xlim([0,200])
                obj.plot_setting(ax, [-30,0], fontsize, 'Arial', 1.25, -30:10:0, 0:50:250, -301:0, 0:10:250)

                if iplot ==1
                    xlabel('Frequency (Hz)')
                    ylabel('Magintude(dBm)')
                else
                    ax.YAxis.TickLabels= [];
                    ax.XAxis.TickLabels= [];
                end
                title(obj.nodeLabel.RegionLabel(i))
            end % end for loop
            sgtitle('\rmLearned filter banks', FontName = 'Arial')
            print -dsvg -vector figures/DS01/DS01_Learned_filterbanks.svg
        end% end function plot_learned_filters
        function plot_MAE(obj,lim)

            MAE_AVG_s = double(smooth(interp1(1:length(obj.MeanAverageError.MAE_AVG),...
                (double(obj.MeanAverageError.MAE_AVG)) ...
                , linspace(1,length(obj.MeanAverageError.MAE_AVG), ...
                length(obj.data_incong_hit_err.data.time)), 'pchip'),10,'sgolay'));

            MAE_VAR_s = smooth(interp1(1:length(obj.MeanAverageError.MAE_VAR),...
                (double(obj.MeanAverageError.MAE_VAR)) ...
                , linspace(1,length(obj.MeanAverageError.MAE_VAR), ...
                length(obj.data_incong_hit_err.data.time)), 'pchip'),10,'sgolay');
            figure
            hold on
            plot(obj.data_incong_hit_err.data.time, MAE_AVG_s, 'LineWidth', 2, 'Color', [76, 199, 100]/255)
            f = fill([obj.data_incong_hit_err.data.time fliplr(obj.data_incong_hit_err.data.time)], ...
                [MAE_AVG_s+sqrt(MAE_VAR_s/double(obj.MeanAverageError.MAE_DF));...
                flipud(MAE_AVG_s-sqrt(MAE_VAR_s/double(obj.MeanAverageError.MAE_DF)))]','g');
            f.EdgeColor = 'none';
            f.FaceAlpha = .3;
            ylabel('Mean absolute error (\fontsize{12}s.e.m\fontsize{16})')
            xlabel('Time(s)')
            hold off
            ax = gca();
            obj.plot_setting(ax, lim, 16, 'Arial', 1.25, lim,...
                -1.5:.5:0.1,[lim(1):.0025:lim(2)],-1.5:.1:0.1);
            xlim([-1.6,.1])
            print -dsvg figures/DS01/DS01_MAE_time.svg
        end % end plot_MAE
        function [m,s, ix] = plot_MAE_chan(obj, lim, aspect, type)
            col = [237, 212, 0
                196, 160, 0
                138, 226, 52
                115, 210, 22
                78, 154, 6
                252, 175, 62
                245, 121, 0
                206, 92, 0
                114, 159, 207
                52, 101, 164
                32, 74, 135
                173, 127, 168
                117, 80, 123
                92, 53, 102
                233, 185, 110
                193, 125, 17
                143, 89, 2
                239, 41, 41
                204, 0, 0]/255;
            %             tval = ((1-double(obj.MeanAverageError_chan.MAE_AVG)) -...
            %                 (1-str2num(obj.accuracy{end})))...
            %                 ./sqrt(double(obj.MeanAverageError_chan.MAE_VAR)./double(obj.MeanAverageError_chan.MAE_DF));

            MAE_AVG_s = (1-double(obj.MeanAverageError_chan.MAE_AVG(:, type)));
            [~,ix]  = sort(MAE_AVG_s);
            MAE_VAR_s =sqrt( double(obj.MeanAverageError_chan.MAE_VAR(:, type))./double(obj.MeanAverageError_chan.MAE_DF));
            figure('Units','normalized',OuterPosition=[0 0 1 1])
            hold on
            c = 20;
            for i = ix'
                c=c-1; % correcting the ascending to descending order
                bar(c, MAE_AVG_s(i),'FaceColor', col(i,:))
                errorbar(c,MAE_AVG_s(i), MAE_VAR_s(i), 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 1.75)

            end

            ylabel('Mean absolute error (\fontsize{18}s.e.m\fontsize{24})')
            xlabel('ROIs')
            hold off
            ax = gca();
            obj.plot_setting(ax,lim, 12, 'Arial Nova Cond', 1.75, lim,...
                1:19,[lim(1):.01:lim(2)],1:19);
            ax.XAxis.TickLabels = obj.nodeLabel.RegionLabel(fliplr(ix'));
            m = MAE_AVG_s(ix);
            s = MAE_VAR_s(ix);
            pbaspect(aspect);
            if type ==1
                print('C:\Users\behira\OneDrive - Karolinska Institutet\Desktop\Manuscript\CognitiveControl\scripts\figures\DS01\DS01_MAE_chan_error.svg', '-dsvg', '-vector')
            else
                print('C:\Users\behira\OneDrive - Karolinska Institutet\Desktop\Manuscript\CognitiveControl\scripts\figures\DS01\DS01_MAE_chan_hit.svg', '-dsvg', '-vector')
            end
        end % end plot_MAE

        function plot_learning_rate(obj, epoch, LR_, decay, decay_steps, FontSize)
            clf
            S = 1:(epoch*190);
            LR = @(step) LR_.*decay.^(floor(step/decay_steps));
            LR_ = @(step) LR_.*decay.^(step/decay_steps);
            plot(S,LR(S), 'Color', [76, 199, 100]/255,'LineWidth', 2)
            hold on
            plot(S,LR_(S), 'k--', 'LineWidth', 2)


            axis square
            box off
            ylabel('Learning rate')
            xlabel('Epochs')
            ax = gca();
            obj.plot_setting(ax,[.8e-4, 2e-4], FontSize, 'Arial', 1.25,  [.8e-4, 2e-4], [0:50*190:(100*190)],...
                [.8e-4:0.2e-4:2e-4],[0:10*190:(100*190)]);
            ax.XAxis.TickLabels = {0:50:100};

            print -dsvg figures\DS01\DS01_learning_rate.svg
        end % plot learning rate
        function [b, p, CI, stat] = apply_filter(obj,index)
            fs = 500;
            c = 0
            for i_index=index
                c = c+1;
                n = length(obj.Kernel_layer.Kernel(:,i_index));
                f = (-n/2:n/2-1)*fs/n;
                FK = fftshift(fft(obj.Kernel_layer.Kernel(:,i_index)));
                b = firls(16,f(end/2+1:end),abs(FK(end/2+1:end))./max(abs(FK(end/2+1:end)))');
                s = squeeze(obj.data_partitioned_prepratory.x_test_r(:,:,i_index));
                ys = obj.data_partitioned_prepratory.y_test_r;
                for i = 1:size(s,1)
                    sf(i,:) = filtfilt(b,1,s(i,:));
                    InsA(i,:) = abs(hilbert(sf(i,:)));
                end

                [~,p{c},CI{c},stat{c}] = ttest2(InsA(ys(:,2) == 1,:),InsA(ys(:,1) == 1,:));
            end

        end
    end % end methods

    methods (Static)
        function plot_setting(ax, ylim_, FontSize, FontName, LineWidth, Ytick, Xtick, YtickM, XtickM)
            axes(ax)
            axis square
            box off
            ax.FontSize = FontSize; % 16
            ax.FontName = FontName; % 'Arial'
            ax.LineWidth= LineWidth;%  1.25
            ax.TickLength(1) = 0.02;

            ax.YAxis.TickValues= Ytick;
            ax.YAxis.MinorTick = 'on';
            ax.YAxis.MinorTickValues = YtickM;%.60:.1:1;
            ax.XAxis.TickValues= Xtick;
            ax.XAxis.MinorTick = 'on';
            ax.XAxis.MinorTickValues= XtickM;
            ylim([ylim_])
        end % plot_setting

        function col = getColor()

            col = [237, 212, 0
                196, 160, 0
                138, 226, 52
                115, 210, 22
                78, 154, 6
                252, 175, 62
                245, 121, 0
                206, 92, 0
                114, 159, 207
                52, 101, 164
                32, 74, 135
                173, 127, 168
                117, 80, 123
                92, 53, 102
                233, 185, 110
                193, 125, 17
                143, 89, 2
                239, 41, 41
                204, 0, 0]/255;
        end
        % ----------------------
        sr2 = plot_grad_cam(Data, path, fpath, load_)
        % -----------------------
    end % methods static
end % class CNNdata
