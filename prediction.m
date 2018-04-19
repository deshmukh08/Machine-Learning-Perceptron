function output = prediction(input,weight_matrix)
    dot_prod = dot_product(input,weight_matrix);
    if(dot_prod >= 0.0)
        output = 1;
    else
        output = 0;
    end
end