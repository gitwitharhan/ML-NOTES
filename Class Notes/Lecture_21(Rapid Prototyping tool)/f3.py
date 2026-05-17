import streamlit as st

st.title("Hello World")
st.header("Class of Agentic AI")
st.subheader("Learning Rapid Prototype Tools")
st.text("abcde")

st.success("Success")
st.warning("Warning")
st.info("Information")
st.error("Error")

#Checkbox
if st.checkbox("Select/Unselect"):
    st.text("User selected the checkbox")
else:
    st.text("User has not selected the checkbox")

#Radio Button
state = st.radio("What is your favorite color?" ,
                 ("Red","Green","Blue"))

if state == "Red":
    st.text("You selected Red")

#SelectBox
occupation = st.selectbox("What do you do?" ,
                          ["Student","Vlogger","Engineer"])
st.text(f"Selected option is {occupation}")

#Button
if st.button("Example Button"):
    st.success("You clicked it")

# st.slider() — interactive range selector
age = st.slider("Select your age", 0, 100, 25)
st.write(f"Your age is: {age}")


# st.code() — syntax-highlighted code display
st.code("""
def greet(name):
    return f"Hello {name}!"

print(greet("World"))
""", language="python")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Column 1")
    st.write("This is the left column.")
    st.image("https://placekitten.com/200/200")

with col2:
    st.header("Column 2")
    st.write("This is the middle column.")
    user_name = st.text_input("Your name")

with col3:
    st.header("Column 3")
    st.write("This is the right column.")
    st.metric("Users", 1200, "+15%")


# ─── st.expander() — collapsible section ───
with st.expander("Click to see more details"):
    st.write("This content is hidden by default!")
    st.write("Users can expand it to read more.")
    st.code("print('Hidden code revealed!')")

with st.expander("Advanced Settings"):
    st.slider("Max Tokens", 100, 4096, 512)
    st.slider("Temperature", 0.0, 2.0, 0.7)
    st.checkbox("Stream output")



#streamlit run f3.py
