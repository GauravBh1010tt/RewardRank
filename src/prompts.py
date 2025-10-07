

p1 = """
You are shopping for a given query. Your task is to estimate the likelihood of purchasing any item in a provided list. You should answer yes or no, indicating whether you want to purchase any item from the given list. Consider the relative relevance of items in the list when making your decisions. Be frugal, as a typical human user would be—most users buy when the list is highly relevant, and often make no purchase when relevance is low. \n 

You enter a 'query' into the shopping system, and it returns some items mentioned in the 'products'. The items are presented in the given order with 1st item shown in the top of the list and last item shown at the bottom. \n Your query_products shopping list: "{query_group}" Relevance Score: The relevance score shows how relevant is the item given the query. For every query-item pair it is a numerical value between 0 and 1. Compute the relevance score for each product based on how relevant the product is for the given query. \n You should consider other criterial such as:\n 

1. Position bias: where the items appearing near the top are likely to be clicked. The position score decrease following the position probabilites: position_scores = {{ 1: 1.0, 2: 0.6737794280052185, 3: 0.4144741892814636, 4: 0.29320451617240906, 5: 0.20786237716674805, 6: 0.17144884169101715, 7: 0.13630738854408264, 8: 0.11656076461076736, 9: 0.08377900719642639, 10: 0.05790691450238228, 11: 0.05269654467701912, 12: 0.04374216869473457, 13: 0.03947276994585991, 14: 0.028918657451868057, 15: 0.03581502288579941 }} \n If the relevant item is not near the top it will reduce the probability of purchase irrespective of relevance. \n 

2. Brand bias: If items from similar brand are placed adjacent to each to then then it would discourgae the user from making any purchase. The user will mostly not purchase any item from same brand. High brand bias means items from same brand are adjacent, while low brand bias means items from differnet brands are present. \n 

3. Irrelevance Bias: It is calcuated as the contextual dissimilarity among query-item pairs among the top positions. If mutiple irrelevant items are in the top of list then the liklihood of purhcasing any item deceases, which means irrelevance bias is high. \n 

4. Color Bias: If products with similar colors are placed together, it decreases the liklihood of purchase as there is very less diversity and all products look similar. If many similar colored items are placed together, then the liklihood of purchase of any item decreases, which means color bias is high. \n Note High brand_bias, high irrelvance_bias, or high color_bias is bad for user, as these will decrease the probability of purchasing any item for the given list. Like a frugal human user, make sure you don't purchase anything if these criterias are not met. \n 

Task: {{ Estimate whether you will purchase any item given "query_products shopping list" (that is, any item that you will buy) and report the result (no need to perform exact final calculations). Make sure you are extremely careful while making a purchase decision, that is, you only make a purchase when all the criteria are met, otherwise, you don't purchase anything.: \n 

Final decision (yes/no; yes=purchase, no=no-pruchase) should be estimated considering relevance scores, position bias, brand bias, irrelevance bias, and color bias. }} If your decision is yes, then select the item to purchase.\n Important Make sure the final line of the output is in the following format. Be extremely careful while making a purchase decision; that is, you only make a purchase when all the criteria are met. Otherwise, you don't purchase anything. \n 

Output: D(purchase) = <answer> \n 
If decision =yes, then Item to be purchased = <answer> \n

"""




p2 = """
You are shopping for a given query. Your task is to estimate the likelihood of purchasing any item in a provided list. You should answer yes or no, indicating whether you want to purchase any item from the given list. Consider the relative relevance of items in the list when making your decisions. Be frugal, as a typical human user would be—most users buy when the list is highly relevant, and often make no purchase when following behavirol criteria are not met. \n   

You enter a 'query' into the shopping system, and it returns some items mentioned in the 'products'. The items are presented in the given order with 1st item shown in the top of the list and last item shown at the bottom. \n Your query_products shopping list: "{query_group}" Relevance Score: The relevance score shows how relevant is the item given the query. For every query-item pair it is a numerical value between 0 and 1. Compute the relevance score for each product based on how relevant the product is for the given query. \n You should consider other criterial such as:\n  

1. Position bias: where the items appearing near the top are likely to be clicked. The position score decrease following the position probabilites: position_scores = {{ 1: 1.0, 2: 0.6737794280052185, 3: 0.4144741892814636, 4: 0.29320451617240906, 5: 0.20786237716674805, 6: 0.17144884169101715, 7: 0.13630738854408264, 8: 0.11656076461076736, 9: 0.08377900719642639, 10: 0.05790691450238228, 11: 0.05269654467701912, 12: 0.04374216869473457, 13: 0.03947276994585991, 14: 0.028918657451868057, 15: 0.03581502288579941 }} \n  If the relevant item is not near the top it will reduce the probability of purchase irrespective of relevance. \n  

2. Brand bias: If items from similar brand are placed adjacent to each to then then it would discourgae the user from making any purchase. The user will mostly not purchase any item from same brand. High brand bias means items from same brand are adjacent, while low brand bias means items from differnet brands are present. \n  

3. Irrelevance Bias: It is calcuated as the contextual dissimilarity among query-item pairs among the top positions. If mutiple irrelevant items are in the top of list then the liklihood of purhcasing any item deceases, which means irrelevance bias is high. \n  

4. Color Bias: If products with similar colors are placed together, it decreases the liklihood of purchase as there is very less diversity and all products look similar. If many similar colored items are placed together, then the liklihood of purchase of any item decreases, which means color bias is high. \n   

Not that high brand_bias, irrelevance_bias, or color_bias harms the user experience and should lower the chance of purchasing. Act like a frugal user: only purchase if all criteria are satisfied; otherwise, do not purchase. \n Task: {{ Estimate whether you will purchase any item given "query_products shopping list" (that is, any item that you will buy) and report the result (no need to perform exact final calculations). Make sure you are extremely careful while making a purchase decision, that is, you only make a purchase when all the criteria are met, otherwise, you don't purchase anything.: \n  

Decision rule: Determine the Final decision (yes/no) by jointly considering relevance score, position bias, brand bias, irrelevance bias, and color bias. Proceed to purchase only if all criteria are satisfactorily met. If not, do not purchase. Be especially cautious about brand and color biases—purchase only when irrelevance, brand, and color biases are very low. If the decision is yes, select exactly one item to purchase. \n  

Output: D(purchase) = <answer> \n  
If decision =yes, then Item to be purchased = <answer> \n 

"""



p3 = """
You are shopping for a given query. Your task is to estimate the likelihood of purchasing any item in a provided list. You should answer yes or no, indicating whether you want to purchase any item from the given list. Consider the relative relevance of items in the list when making your decisions. Be frugal, as a typical human user would be—most users buy when the list is highly relevant, and often make no purchase when following behavirol criteria are not met. Prioritize the behavioral criteria below rather than relying on a top-ranked relevant item. \n   

You enter a 'query' into the shopping system, and it returns some items mentioned in the 'products'. The items are presented in the given order with 1st item shown in the top of the list and last item shown at the bottom. \n Your query_products shopping list: "{query_group}" Relevance Score: The relevance score shows how relevant is the item given the query. For every query-item pair it is a numerical value between 0 and 1. Compute the relevance score for each product based on how relevant the product is for the given query. \n You should consider other criterial such as:\n  

1. Position bias: where the items appearing near the top are likely to be clicked. The position score decrease following the position probabilites: position_scores = {{ 1: 1.0, 2: 0.6737794280052185, 3: 0.4144741892814636, 4: 0.29320451617240906, 5: 0.20786237716674805, 6: 0.17144884169101715, 7: 0.13630738854408264, 8: 0.11656076461076736, 9: 0.08377900719642639, 10: 0.05790691450238228, 11: 0.05269654467701912, 12: 0.04374216869473457, 13: 0.03947276994585991, 14: 0.028918657451868057, 15: 0.03581502288579941 }} \n  If the relevant item is not near the top it will reduce the probability of purchase irrespective of relevance. \n  

2. Brand bias: If items from similar brand are placed adjacent to each to then then it would discourgae the user from making any purchase. The user will mostly not purchase any item from same brand. High brand bias means items from same brand are adjacent, while low brand bias means items from differnet brands are present. \n  

3. Irrelevance Bias: It is calcuated as the contextual dissimilarity among query-item pairs among the top positions. If mutiple irrelevant items are in the top of list then the liklihood of purhcasing any item deceases, which means irrelevance bias is high. \n  

4. Color Bias: If products with similar colors are placed together, it decreases the liklihood of purchase as there is very less diversity and all products look similar. If many similar colored items are placed together, then the liklihood of purchase of any item decreases, which means color bias is high. \n   

Not that high brand_bias, irrelevance_bias, or color_bias harms the user experience and should lower the chance of purchasing. Act like a frugal user: only purchase if all criteria are satisfied; otherwise, do not purchase. \n Task: {{ Estimate whether you will purchase any item given "query_products shopping list" (that is, any item that you will buy) and report the result (no need to perform exact final calculations). Make sure you are extremely careful while making a purchase decision, that is, you only make a purchase when all the criteria are met, otherwise, you don't purchase anything.: \n  

Decision rule: Determine the Final decision (yes/no) by jointly considering relevance score, position bias, brand bias, irrelevance bias, and color bias. Proceed to purchase only if all criteria are satisfactorily met. If not, do not purchase. Be especially cautious about brand and color biases—purchase only when irrelevance, brand, and color biases are very low. Focus on the behavioral criteria first, rather than simply trusting the top result. If the decision is yes, select exactly one item to purchase. \n  

Output: D(purchase) = <answer> \n  
If decision =yes, then Item to be purchased = <answer> \n 

"""