import pandas as pd

# Load the CSV file
df = pd.read_csv('backtest_results/20251129_021831/a2a_allocations.csv')

# Assuming the first column is timestamp, extract allocation columns
allocations = df.iloc[:, 1:]

print("Data shape:", allocations.shape)
print("Columns:", list(allocations.columns)[:10], "...")  # Show first 10 columns

# 1. Number of non-zeros per row
non_zeros = allocations.apply(lambda row: (row != 0).sum(), axis=1)
print("\n1. Number of non-zeros per row:")
print(non_zeros.describe())

# 2. Maximum element per row
max_vals = allocations.max(axis=1)
print("\n2. Maximum element per row:")
print(max_vals.describe())

# 3. Sum of weights per row
sums = allocations.sum(axis=1)
print("\n3. Sum of weights per row:")
print(sums.describe())

# 4. Number of weights that change more than 100% from one row to the next
changes = []
for i in range(len(allocations) - 1):
    row1 = allocations.iloc[i]
    row2 = allocations.iloc[i + 1]
    count = 0
    for col in allocations.columns:
        old = row1[col]
        new = row2[col]
        if old != 0:
            pct_change = abs(new - old) / abs(old)
            if pct_change > 1:
                count += 1
        elif new != 0:
            # Consider it as a change if it goes from 0 to non-zero
            count += 1
    changes.append(count)

changes_series = pd.Series(changes)
print("\n4. Number of weights that change more than 100% from one row to the next:")
print(changes_series.describe())