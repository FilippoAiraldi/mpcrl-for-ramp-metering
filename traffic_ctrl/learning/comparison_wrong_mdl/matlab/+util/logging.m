function logging(k, K, simtime, realclock, msg)
    if nargin == 4
        msg = '';
    else
        msg = append(' - ', msg);
    end
    d_real = duration(0, 0, realclock, 'Format', 'hh:mm:ss');
    d_sim = duration(simtime, 0, 0, 'Format', 'hh:mm:ss');
    fmt = ['[%s|%', int2str(ceil(log10(K))), 'i|%04.1f%%|%s]%s\n'];
    fprintf(fmt, d_real, k, k / K * 100, d_sim, msg);
end