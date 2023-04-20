import os
from typing import Optional, Union, Dict, Any
import logging
import re

import langchain
from langchain.tools import BaseTool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from arxiv_bot import templates
from arxiv_bot.knowledge_base.database import Pinecone
from arxiv_bot.knowledge_base.constructors import Arxiv, init_extractor, get_paper_id

paper_id_re = re.compile(r'(\d{4}\.\d{4,6})')

class AddArticleTool(BaseTool):
    name = "Add to ArXiv DB"
    description = "use this tool to add a new paper to the ArXiv DB. If human provides ArXiv ID like 1012.1313 pass this to the tool. Otherwise try and use the human's query."
    memory: Any = None

    def _run(self, query: str) -> str:
        """Function to be used for adding articles to the ArXiv database it expects
        an arXiv ID as input and will add the article to the database.
        """
        # get ArXiv ID from input text if exists
        # extract the paper id
        matches = paper_id_re.findall(query)
        if matches == []:
            # check we have 'arxiv' and 'paper' terms in query
            for term in ['arxiv', 'paper']:
                if term not in query.lower():
                    # add the terms to the query if not
                    query = ' '.join([query, term])
            # try and extract if this is natural language query
            paper_id = get_paper_id(query)
        else:
            paper_id = matches[0]
        # check again for paper ID
        if paper_id is None:
            return "ArXiv paper could not be found"
        # get single arxiv object
        paper = Arxiv(paper_id)
        # load the paper
        paper.load()
        # get paper metadata
        paper_metadata = paper.get_meta()
        # chunk into smaller parts
        paper.chunker()
        # now add to the database
        ids = []
        texts = []
        metadatas = []
        for record in paper.dataset:
            record = {**record, **paper_metadata}
            ids.append(f"{record['id']}-{record['chunk-id']}")
            texts.append(record['chunk'])
            for feature in ['id', 'chunk-id', 'summary', 'authors', 'comment', 'categories', 'journal_ref', 'references', 'doi', 'chunk']:
                record.pop(feature)
            metadatas.append(record)
        # add to the database
        self.memory.add(texts, ids=ids, metadata=metadatas)
        return f"Added {paper_id} to the memory. It is now accessible."

    def _arun(self, query: str) -> str:
        raise NotImplementedError("AddArticleTool does not support async")

class Arxiver:
    search_description = (
        "Use this tool when searching for scientific research information "
        "from our prebuilt ArXiv papers database. This should be the first "
        "option when looking for information. When recieving information from "
        "this tool you MUST always include all sources of information."
    )
    sys_msg = (
        "You are an expert summarizer and deliverer of technical information. "
        "Yet, the reason you are so intelligent is that you make complex "
        "information incredibly simple to understand. It's actually rather "
        "incredible. When users ask information you refer to the relevant tools "
        "when needed, allowing you to answer questions about a broad range of "
        "technical topics.\n"
        "When external information is used you MUST add your sources to the end "
        "of responses."
    )
    ref_template = templates.reference_extraction['template']
    tools = []
    def __init__(
        self,
        index_name: str = "arxiv-bot",
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        verbose: bool = False
    ):
        # first we initialize the LLM
        self._init_llm(openai_api_key=openai_api_key)
        # next initialize memory retrieval
        self._init_memory(
            index_name=index_name,
            pinecone_api_key=pinecone_api_key,
            pinecone_environment=pinecone_environment
        )
        # initialize the chatbot
        self._init_chatbot(verbose=verbose)
        # initialize the extraction tooling
        self._init_splitter()

    def __call__(self, text: str, detailed: bool = False) -> dict:
        response = self.agent(text)
        if detailed:
            return response
        else:
            return {'output': response['output']}
        
    def _init_splitter(self):
        self.extractor, self.text_splitter = init_extractor(self.ref_template)

    def _init_llm(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo"
    ):
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.llm = langchain.chat_models.ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0,
            model_name=model_name
        )

    def _init_memory(
        self,
        index_name: str = "arxiv-bot",
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        pinecone_environment = pinecone_environment or os.environ.get("PINECONE_ENVIRONMENT")
        openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        # initialize memory
        self.memory = Pinecone(
            index_name=index_name,
            pinecone_api_key=pinecone_api_key,
            environment=pinecone_environment,
            openai_api_key=openai_api_key
        )
        # initialize embedding model
        self.encoder = langchain.embeddings.openai.OpenAIEmbeddings(
            openai_api_key=openai_api_key
        )
        # initialize vector db
        self.vectordb = langchain.vectorstores.Pinecone(
            index=self.memory.index,
            embedding_function=self.encoder.embed_query,
            text_key='text-field'
        )
        # initialize retriever
        self.retriever = langchain.chains.RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vectordb.as_retriever()
        )
        # initialize search tool
        arxiv_db_tool = langchain.agents.Tool(
            func=self._search_arxiv_db,
            description=self.search_description,
            name="Search ArXiv DB"
        )
        # append to tools list
        self.tools.append(arxiv_db_tool)
        # initialize add article tool
        article_add_tool = AddArticleTool()
        article_add_tool.memory = self.memory
        # append to tools list
        self.tools.append(article_add_tool)

    def _init_chatbot(self, verbose: bool = False):
        # initialize conversational memory
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        # initialize agent with tools
        self.agent = langchain.agents.initialize_agent(
            agent='chat-conversational-react-description',
            tools=self.tools,
            llm=self.llm,
            verbose=verbose,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversational_memory
        )
        # then update the prompt
        prompt = self.agent.agent.create_prompt(
            system_message=self.sys_msg,
            tools=self.tools
        )
        self.agent.agent.llm_chain.prompt = prompt

    def _add_article_to_db(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False
    ) -> Dict[str, Any]:
        """Function to be used for adding articles to the ArXiv database it expects
        an arXiv ID as input and will add the article to the database.
        """
        arxiv_doi = inputs['arxiv_doi']
        # get single arxiv object
        paper = Arxiv(arxiv_doi)
        # load the paper
        paper.load()
        # get paper metadata
        paper_metadata = paper.get_meta()
        # chunk into smaller parts
        paper.chunker()
        # now add to the database
        ids = []
        texts = []
        metadatas = []
        for record in paper.dataset:
            record = {**record, **paper_metadata}
            ids.append(f"{record['id']}-{record['chunk-id']}")
            texts.append(record['chunk'])
            for feature in ['id', 'chunk-id', 'summary', 'authors', 'comment', 'categories', 'journal_ref', 'references', 'doi', 'chunk']:
                record.pop(feature)
            metadatas.append(record)
        # add to the database
        self.memory.add(texts, ids=ids, metadata=metadatas)
        return {'answer': f"Added {arxiv_doi} to my memory. It is now accessible via the Search Arxiv DB tool."}


    def _search_arxiv_db(
        self,
        inputs: Union[Dict[str, Any], Any],
        return_only_outputs: bool = False
    ) -> Dict[str, Any]:
        """Custom search function to be used for ArXiv retrieval tools. Modifies the
        typical langchain retrieval function by adding the sources to the answer.
        
        :params:
            inputs: Union[Dict[str, Any], Any] - inputs to the search function in
                the form of a dictionary like {'question': 'some question'}
            return_only_outputs: bool - whether to return only the outputs or the
                entire dictionary of inputs and outputs
        :returns:
            outputs: Dict[str, Any] - outputs from the search function in the form of
                a dictionary like {'answer': 'some answer', 'sources': 'some sources'}
        """
        inputs = self.retriever.prep_inputs(inputs)
        self.retriever.callback_manager.on_chain_start(
            {"name": self.retriever.__class__.__name__},
            inputs,
            verbose=self.retriever.verbose,
        )
        try:
            outputs = self.retriever._call(inputs)
            # add the sources to the 'answer' value
            outputs['answer'] = outputs['answer'].replace('\n', ' ') + ' - sources: ' + outputs['sources']
        except (KeyboardInterrupt, Exception) as e:
            self.retriever.callback_manager.on_chain_error(e, verbose=self.retriever.verbose)
            raise e
        self.retriever.callback_manager.on_chain_end(outputs, verbose=self.retriever.verbose)
        return self.retriever.prep_outputs(inputs, outputs, return_only_outputs)
