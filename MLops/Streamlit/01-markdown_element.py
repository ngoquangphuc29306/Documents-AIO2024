import streamlit as st

# Text Element
st.title("This is a title")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is a text")

# Markdown element
st.markdown("# Peter Phuc")
st.markdown("## Peter Phuc")
st.markdown("### Peter Phuc")

st.markdown("**Peter Phuc**")
st.markdown("*Peter* Phuc")

st.markdown("> Peter Phuc")
st.markdown("1. First Item\n 2. Second Item")
str = "print('hello world')"
st.code(str)
st.markdown("---")
st.markdown("[Google](https://www.google.com/)")

table = '''
| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |
'''
st.markdown(table)

json = {
  "firstName": "Peter",
  "lastName": "Phuc",
  "age": 25
}
st.json(json)

st.markdown('That is so funny! :joy:')
