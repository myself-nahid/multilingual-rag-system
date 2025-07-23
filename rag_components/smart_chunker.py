from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.document import Document
import config

from pydantic import BaseModel, Field
from typing import List


class Paragraph(BaseModel):
    paragraph_text: str = Field(
        description="A single, complete, and self-contained paragraph from the document.")


class ParagraphList(BaseModel):
    paragraphs: List[Paragraph] = Field(
        description="A list of all reconstructed paragraphs from the provided page text.")


def get_semantic_chunks_from_page(page_text: str, page_number: int) -> List[Document]:
    """
    Uses an LLM to intelligently split a raw page text into semantic paragraphs.
    """
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0)

    parser = JsonOutputParser(pydantic_object=ParagraphList)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert document analysis assistant. Your task is to analyze the raw text extracted from a PDF page and reconstruct the original paragraphs.
        The text may have incorrect line breaks or formatting issues. Ignore them and focus on creating semantically coherent paragraphs.
        
        Return the paragraphs as a JSON list. Do not include headers, footers, or page numbers as separate paragraphs.

        {format_instructions}
        
        Page Text:
        ```{text}```
        """
    )

    chain = prompt | llm | parser

    try:
        response = chain.invoke({
            "text": page_text,
            "format_instructions": parser.get_format_instructions(),
        })

        documents = []
        for para in response['paragraphs']:
            doc = Document(
                page_content=para['paragraph_text'],
                metadata={
                    "source_page": page_number
                }
            )
            documents.append(doc)
        return documents

    except Exception as e:
        print(
            f"--- Failed to parse page {page_number} with LLM. Returning as a single chunk. Error: {e} ---")
        return [Document(page_content=page_text, metadata={"source_page": page_number})]
