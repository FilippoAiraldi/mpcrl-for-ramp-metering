function lr = onecycle_lr(epoch, epochs_max, lr_max)
    cut = ceil(epochs_max * 0.3);
    if epoch < cut
        lr = cos_anneal(epoch, cut, lr_max / 25, lr_max);
    else
        lr = cos_anneal(epoch - cut, epochs_max - cut, lr_max, lr_max / 25e3);
    end
end

function y = cos_anneal(i, T, lr_start, lr_end)
    y = lr_end + 0.5 * (lr_start - lr_end) * (1 + cos(i * pi / T));
end

