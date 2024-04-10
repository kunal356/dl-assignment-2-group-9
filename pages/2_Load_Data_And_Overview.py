import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def display_graphs(df):
    df = df.sample(1000)
    st.write(df.shape)
    all_columns_names = df.columns.to_list()
    selected_column_names = st.multiselect("Select Columns to plot", all_columns_names)
    cust_data = df[selected_column_names]
    st.area_chart(cust_data)
    st.bar_chart(cust_data)
    st.line_chart(cust_data)

    fig, ax = plt.subplots()
    p1 = cust_data.plot(kind='hist', ax=ax)
    st.write(p1)
    st.pyplot(fig)
    
    fig1, ax1 = plt.subplots()
    p2 = cust_data.plot(kind='box', ax=ax1)
    st.write(p2)
    st.pyplot(fig1)
    

def main():
    st.title("Load Data and Overview")
    data = st.file_uploader("Upload your Dataset", type=['csv'])
    if data is not None:
        df = pd.read_csv(data)
        st.write(df.head())
        display_graphs(df)
if __name__ == "__main__":
    main()

