import matplotlib.pyplot as plt
import pandas as pd


results_df = pd.read_csv('model_comparison_results.csv')

plt.figure(figsize=(14, 8))

for model_name in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model_name]
    plt.plot(subset['Visual Parameter'], subset['Mean Squared Error'], label=model_name, marker='o')

plt.xlabel('Visual Parameter')
plt.ylabel('Mean Squared Error')
plt.title('Model Comparison by Mean Squared Error for Each Visual Parameter')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
