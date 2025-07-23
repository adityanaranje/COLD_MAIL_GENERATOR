import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
        model = "llama3-70b-8192",
        groq_api_key = GROQ_API_KEY,
        temperature=0
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION
            The scraped text is from the careers's page of a website.
            Your job is to extract the job postings and return them in json format containing following keys: `role`, `experience`,`skills` and `description`
            Only return the valid json NO PREAMBLE.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data":cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs")
        return res if isinstance(res, list) else [res]
    
    def write_email(self, jobs, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION
            {job_description}
            ### INSTRUCTION:
                You are Aditya, a business analyst at Nvest Solutions. Nvest Solution is a financial technology firm building tools to improve the efficiency in insurance and financial sector through various technologies. 
                Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                process optimization, cost reduction, and heightened overall efficiency. 
                Your job is to write a cold email to the client regarding the job mentioned above describing the capability of yourself 
                in fulfilling their needs.
                Also add the most relevant ones from the following links to showcase your portfolio: {link_list}
                Remember you are Aditya, BA at Nvest Solutions. 
                Do not provide a preamble.
                ### EMAIL (NO PREAMBLE):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description":str(jobs), "link_list":links})
        return res.content


