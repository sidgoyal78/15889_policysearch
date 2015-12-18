function a = choose_next_action(theta, cur)
    if cur == 1
      %  p1 = [1 0 0 0];
      %  p2 = [0 0 1 0];
      %  val = exp([p1*theta , p2*theta]);
      %  val = val/ sum(val);
       
       rolldice = unifrnd(0, 1);
      %   val = val(1);
         val = theta(1);
      
       if rolldice < val
             a = 1;
        else
            a = 2;
        end
      %  [m,n] = max(val);
      %  a = n;
    else
       % p1 = [0 1 0 0];
       % p2 = [0 0 0 1];
       % val = exp([p1*theta , p2*theta]);
       % val = val / sum(val);
       % val = val(1);
        
        val = theta(2);
        rolldice = unifrnd(0,1);
        if rolldice < val
            a = 1;
        else
            a = 2;
        end
        %[m,n] = max(val);
        %a = n;
    end
end