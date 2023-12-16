from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    folder_path = 'csv/'
    fname1 = folder_path + 'Monday-WorkingHours.pcap_ISCX.csv'
    fname2 = folder_path + 'Tuesday-WorkingHours.pcap_ISCX.csv'
    fname3 = folder_path + 'Wednesday-workingHours.pcap_ISCX.csv'
    fname4 = folder_path + 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    fname5 = folder_path + 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    fname6 = folder_path + 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
    fname7 = folder_path + 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    fname8 = folder_path + 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    df1, df2, df3, df4, df5, df6, df7, df8 = load_data(fname1), load_data(fname2), load_data(fname3), load_data(fname4), load_data(fname5), load_data(fname6), load_data(fname7), load_data(fname8)
    
    df_list = [df1, df2, df3, df4, df5, df6, df7, df8]
    dfs = copy.deepcopy(df_list)
    for df in dfs:
        clean_data(df)
        
    df = pd.concat(dfs[1:])
    df.to_csv(path_or_buf=folder_path+'clean_traffic_data.csv' , index=False) # output to one csv
    
    df = load_data(folder_path+'clean_traffic_data.csv')
    
    # DTREE
    # dtree for each file?
    
    # KNN
    # whole dataset for knn?
    
    # MLP
    # resample data
    
    # RF
    # resample data
    
    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
    disp.plot(colorbar=False)
    plt.xticks(rotation=45, ha='right')
    plt.show()
    