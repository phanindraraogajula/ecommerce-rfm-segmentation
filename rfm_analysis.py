import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================
print("Loading data...")
df = pd.read_csv('online_retail_II.csv', encoding='unicode_escape')
print(f"Loaded {len(df)} rows")
print(df.head())

# ============================================================
# STEP 2: CLEAN THE DATA
# ============================================================
print("\nCleaning data...")

# Remove cancelled orders (Invoice starts with 'C')
df = df[~df['Invoice'].astype(str).str.startswith('C')]

# Remove rows with no Customer ID
df = df.dropna(subset=['Customer ID'])

# Remove negative or zero quantities and prices
df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

# Calculate total amount per row
df['TotalAmount'] = df['Quantity'] * df['Price']

print(f"Clean data: {len(df)} rows remaining")

# ============================================================
# STEP 3: RFM CALCULATION
# ============================================================
print("\nCalculating RFM scores...")

# Set reference date (day after last purchase in dataset)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('Customer ID').agg(
    Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency = ('Invoice', 'nunique'),
    Monetary  = ('TotalAmount', 'sum')
).reset_index()

print(rfm.head())

# ============================================================
# STEP 4: SCORE EACH CUSTOMER 1-5
# ============================================================

# Recency: lower days = better = score 5
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1]).astype(int)

# Frequency: higher = better = score 5
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)

# Monetary: higher = better = score 5
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)

# Combined RFM score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# ============================================================
# STEP 5: LABEL CUSTOMER SEGMENTS
# ============================================================

def assign_segment(row):
    r = row['R_Score']
    f = row['F_Score']
    m = row['M_Score']

    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r >= 3 and f <= 2:
        return 'Potential Loyalists'
    elif r == 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2 and m >= 3:
        return 'Cant Lose Them'
    elif r == 1 and f >= 3:
        return 'Lost Champions'
    else:
        return 'Lost'

rfm['Segment'] = rfm.apply(assign_segment, axis=1)

print("\nSegment breakdown:")
print(rfm['Segment'].value_counts())

# ============================================================
# STEP 6: EXPORT TO CSV FOR TABLEAU
# ============================================================
rfm.to_csv('rfm_segments.csv', index=False)
print("\n✅ Saved: rfm_segments.csv")

# ============================================================
# STEP 7: QUICK CHARTS TO PREVIEW
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Customer count per segment
segment_counts = rfm['Segment'].value_counts()
axes[0].bar(segment_counts.index, segment_counts.values, color='steelblue')
axes[0].set_title('Customer Count by Segment')
axes[0].set_xlabel('Segment')
axes[0].set_ylabel('Number of Customers')
axes[0].tick_params(axis='x', rotation=45)

# Chart 2: Revenue per segment
segment_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
axes[1].bar(segment_revenue.index, segment_revenue.values, color='coral')
axes[1].set_title('Total Revenue by Segment')
axes[1].set_xlabel('Segment')
axes[1].set_ylabel('Revenue (£)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('rfm_charts.png', dpi=150)
plt.show()
print("✅ Saved: rfm_charts.png")

print("\n🎉 RFM Analysis Complete!")
print(f"Total customers analyzed: {len(rfm)}")
print(f"Segments created: {rfm['Segment'].nunique()}")