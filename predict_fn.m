function predict_fn(images_train,weights)

%[activation,~] = weights(0)

%for i=1 : 1000
   activation = activation + weight * images_train(i); 
   
   if (activation >= 0)
       prediction = 1;
   else
       prediction = 0;
       
   end
    
end