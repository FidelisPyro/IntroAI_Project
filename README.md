# introAI_project

Here is the true start of the Ant Colony Optimization.  
I have a csv file "generator_cost_emission_coefficients.csv" that has the generator info.  

I have ran the code three times and the best result I have so far is:  
Best Solution: [176, 51, 16, 17, 12, 12]  
Best cost: 771.78776  

This can still be optimized more, the paper "Optimal Solution fo Economic Load Dispatch  
using Teaching Learning Algorithm" got it down to 767.6021  
  
That is a different optimization technique, but I haven't messed around with parameters  
so I'm sure ours can be dropped lower.  
  
This is also only running it to find the minimum cost for the load demand 283.5 MW  
which is the sum of P(MW) column for all 30 buses. One thing to improve this would be  
to make a set this runs through of different load demands from 117 (the sum of the   
minimuns the generators can output) to 283.5.   

I also want to get some graphs using matplotlib to show these current results, and  
any future results we get after more optimizations. 
