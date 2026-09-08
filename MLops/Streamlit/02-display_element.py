import streamlit as st
import pandas as pd

table = ({"Column 1":[1, 2,3,4,5],
          "Column 2":[6,7,8,9,10]})

st.table(table)
st.dataframe(table)
st.metric(label='Win Speed',
          value='70ms', delta='-2.8')