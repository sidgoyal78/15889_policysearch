function a = evaluate_and_estimate(pol, E,T)
    lor = 0.0;
    for i=1:E
        cs = randi(5);
        %cs = 2;
        totalr = 0.0;
        for j = 1:T
            action = choose_next_action(pol, cs);
            ret = next_state(cs, action);
            totalr = totalr + ret(1);
            cs = ret(2);
        end
        lor = lor + (totalr/T);
        
    end
    a = lor / E;
end


