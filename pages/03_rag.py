from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
from langchain.vectorstores import Weaviate
from langchain_community.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import tempfile


def main():
    st.set_page_config(page_title="Ask your PDF")
    load_dotenv()
    YOUR_OPENAI_KEY = os.getenv("YOUR_OPENAI_KEY")
    YOUR_WEAVIATE_KEY = os.getenv("YOUR_WEAVIATE_KEY")
    YOUR_WEAVIATE_CLUSTER = os.getenv("YOUR_WEAVIATE_CLUSTER")
    YOUR_HUGGINGFACE_APIKEY = os.getenv("YOUR_HUGGINGFACE_APIKEY")
    model = Ollama(model="llama3")
    embeddings = OllamaEmbeddings(model="llama3")

    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf.read())
            temp_file_path = tmp_file.name

        # Use PyPDFLoader with the temporary file path
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=30
        )
        docs = text_splitter.split_documents(data)

        # connect Weaviate Cluster
        auth_config = weaviate.AuthApiKey(api_key=YOUR_WEAVIATE_KEY)
        client = weaviate.Client(
            url=YOUR_WEAVIATE_CLUSTER,
            additional_headers={"X-HuggingFace-Api-Key": YOUR_HUGGINGFACE_APIKEY},
            auth_client_secret=auth_config,
        )
        client.schema.delete_all()
        client.schema.get()
        schema = {
            "classes": [
                {
                    "class": "Chatbot",
                    "description": "Documents for chatbot",
                    "vectorizer": "text2vec-huggingface",
                    "moduleConfig": {
                        "text2vec-huggingface": {
                            "model": "sentence-transformers/all-MiniLM-L6-v2",
                            "type": "text",
                        }
                    },
                    "properties": [
                        {
                            "dataType": ["text"],
                            "description": "The content of the paragraph",
                            "moduleConfig": {
                                "text2vec-huggingface": {
                                    "skip": False,
                                    "vectorizePropertyName": False,
                                }
                            },
                            "name": "content",
                        },
                    ],
                },
            ]
        }

        client.schema.create(schema)
        vectorstore = Weaviate(client, "Chatbot", "content", attributes=["source"])
        text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
        texts, meta = list(zip(*text_meta_pair))
        vectorstore.add_texts(texts, meta)

        parser = StrOutputParser()
        template = """
        Answer the question based on the context below. If you can't answer the question, reply "I don't know"
        Context: {context}
        Question: {question}
        """

        prompt = PromptTemplate.from_template(template)
        retriever = vectorstore.as_retriever()
        from operator import itemgetter

        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | model
            | parser
        )
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            response = chain.invoke({"question": user_question})
            st.write(response)

        # Clean up temporary file
        os.remove(temp_file_path)


if __name__ == "__main__":
    main()
