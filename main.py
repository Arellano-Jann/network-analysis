def main():
    
    df = load_data('clean_traffic_data.csv')
    for df in dfs:
        clean_data(df)
    # SAME AS ABOVE
    # for i in range(len(dfs)):
    #     dfs[i].replace([np.inf, -np.inf], np.nan, inplace=True)
    #     dfs[i] = dfs[i].fillna(0) # or df = df.dropna()