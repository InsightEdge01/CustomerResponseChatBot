import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.llms import Replicate
from dotenv import load_dotenv

load_dotenv()

# vectorise the sales response csv data
loader = CSVLoader(file_path="sales_response_data.csv")
documents = loader.load()

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})
db = FAISS.from_documents(documents, embeddings)

# function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents = [doc.page_content for doc in similar_response]
    return page_contents

template = """
You are a world class business development representative. I will share a prospect's
message with you and you will give me the best answer that I should send to this prospect
 based on their responses, and you will follow ALL of the rules below:
 1/ Response should be very similar or even identical to the past responses,
 in terms of length, tone of voice, logical arguments and other details
 
 2/ if the responses are irrelevant, then try to mimic the style of the 
 best practice
 
 below is a message I receive from the prospect:
 {message}
 
 Here is a list of responses of how we normally respond to prospect in similar scenario
 {prospect_responses}
 
 please write the best response that I should send to this prospect:
 """
prompt = PromptTemplate(
    input_variables=['message', 'prospect_responses'],
    template=template
)

#llm = CTransformers(
        #model = "llama-2-7b-chat.ggmlv3.q4_0.bin",
       #model_type="llama",
       #max_new_tokens = 512,
    #temperature = 0.5)

llm = Replicate(
    model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
    input={"temperature": 0.01, "max_length": 500, "top_p": 1})

chain = LLMChain(llm=llm, prompt=prompt)


# retrieval augmented generation
def generate_response(message):
    prospect_responses = retrieve_info(message)
    response = chain.run(message=message, prospect_responses=prospect_responses)
    return response


# build an app with streamlit
def main():
    #st.set_page_config(
        #page_title="Customer Response generator", page_icon=":books")

    st.header("Customer Response Generator :books:")
    message = st.text_area("Customer Message")
    response = st.empty()

    if st.button("Send"):
        if message:
            response.write("Generating message....")
            result = generate_response(message)
            response.info(result)


if __name__ == '__main__':
    main()
