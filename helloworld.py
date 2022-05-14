import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

with st.echo(code_location='below'):
    def print_hello(name = "Гость"):
        st.write(f"## Привет, {name}!")

    name = st.text_input("Ваше имя", key='name', value= "Гость")
    print_hello(name)
