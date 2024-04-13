import streamlit as st
def main():
    st.write("Train And Deploy My DNNs")
    selected_model = st.selectbox(label="Select my DNN",options=["DNN 1", "DNN 2", "DNN3"])
    print(selected_model)
    

if __name__ == "__main__":
    main()