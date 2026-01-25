# pip install mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------
# 1. Create transaction data
# ----------------------------
transactions = [
    ["Milk", "Bread", "Butter"], 
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Butter"],
    ["Bread", "Butter"]
]

# ----------------------------
# 2. One-hot encode transactions
# (Required for Apriori)
# ----------------------------
# Create an encoder object to look at all transactions and find which unique items exist
# e.g. from our data, we will get ["Bread", "Butter", "Milk"]
te = TransactionEncoder()

# te.fit(transactions) -> Scans all transactions and learns vocabulary of terms, i.e. Bread, Butter, Milk
# .transform(transactions) -> Converts each transaction into binary format
# Example: First row will become [True, True, True], second will be [True, True, False]
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)
print("One-hot encoded transaction data:\n")
print(df)

# ----------------------------
# 3. Apply Apriori algorithm
# ----------------------------
frequent_itemsets = apriori(
    df,
    min_support=0.4,   # itemset must appear in at least 40% transactions, i.e. in 2 transactions out of 5
    use_colnames=True # Instead of 0, 1, 2 use Bread, Butter, Milk
)
print(frequent_itemsets)

'''
Sample output
   support        itemsets
0     0.8         (Bread)       -> Meaning: Bread appears in 80% transactions and is frequent 1-itemset
1     0.6        (Butter)
2     0.6           (Milk)
3     0.4   (Bread, Butter)
4     0.4     (Bread, Milk)
'''

# ----------------------------
# 4. Generate association rules
# ----------------------------
# Converts frequent itemsets into IF–THEN rules and keeps only the strong ones.
# Each rule has: IF (antecedents) → THEN (consequents)
# Example: {Bread} → {Butter}
# Suppose support(Bread) = 0.8, support(Bread, Butter) = 0.4
# Confidence formula: confidence(A → B) = support(A ∪ B) / support(A)
# Meaning: Given A occurred, how often did B also occur?
# confidence(Bread → Butter) = 0.4 / 0.8 = 0.5 (discarded - see below)
# confidence(Butter → Bread) = 0.4 / 0.6 ≈ 0.67 (kept)
rules = association_rules(
    frequent_itemsets, # Input to the rule generation, only passing minimum support
    metric="confidence", # Use confidence to decide if a rule is strong
    min_threshold=0.6 # Minimum confidence must be 60%, i.e. rule must be correct at least 60% of the time
)

print("\nAssociation Rules:\n")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
