function kv = kernelf(x, y)
    kv = exp(-0.5  * (x - y) * (x - y)' / (100.0)^2);
end