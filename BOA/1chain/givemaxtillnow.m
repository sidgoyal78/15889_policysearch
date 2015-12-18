function b = givemaxtillnow(a)
    b = [];
    for i = 1:size(a)
        b = [b; max(a(1:i))];
    end
end