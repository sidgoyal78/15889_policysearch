function a = choose_next_action_mountaincar(theta, cur)
    p1 = horzcat(cur, 0*cur, 0*cur);
    p2 = horzcat(0*cur, 1*cur, 0*cur);
    p3 = horzcat(0*cur, 0*cur, 1*cur);
    
    val = exp([p1 * theta, p2 * theta, p3 * theta]);
    val = val / sum(val);
    rollit = rand;
    a = min(find(rollit <= cumsum(val)));
end
