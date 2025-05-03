from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

def build_rag_qa_chain(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def answer_question(question):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = prompt.invoke({"context": context, "question": question})
        return llm.invoke(final_prompt).content

    return answer_question
