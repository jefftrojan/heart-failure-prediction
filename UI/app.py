import streamlit as st

# Set the title of the app
st.title("My Streamlit App")

# Create a header
st.header("Welcome to my app!")

# Create a text input field
name = st.text_input("What is your name?")

# Create a button
if st.button("Submit"):
    # Display a greeting when the button is clicked
    st.write(f"Hello, {name}!")

# Create a selectbox
options = ["Option 1", "Option 2", "Option 3"]
selected_option = st.selectbox("Choose an option", options)

# Display the selected option
st.write(f"You selected: {selected_option}")

# Create a slider
level = st.slider("Select a level", 0, 100, 50)

# Display the selected level
st.write(f"Level: {level}")

# Create a checkbox
checked = st.checkbox("Check me!")

# Display a message when the checkbox is checked
if checked:
    st.write("You checked the box!")