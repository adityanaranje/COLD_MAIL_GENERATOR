import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from portfolios import Portfolio
from utils import clean_text

def create_steramlit_app(llm, portfolio, clean_text):
    st.title("COLD EMAIL GENERATOR")
    url_inputs = st.text_input("ENTER A URL:", value = "https://careers.nike.com/software-engineer-ii-itc/job/R-65543")
    submit_button = st.button("Generate")

    if submit_button:
        try:
            loader = WebBaseLoader([url_inputs])
            data = clean_text(loader.load().pop().page_content)
            portfolio.load_portfolio()
            jobs = llm.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills',[])
                links = portfolio.query_links(skills)
                email = llm.write_email(job, links)
                st.code(email, language='markdown')
        except Exception as e:
            st.error(f"An Error Occured: {e}")

if __name__ == "__main__":
    chain = Chain()
    portfolio = Portfolio()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_steramlit_app(chain, portfolio, clean_text)