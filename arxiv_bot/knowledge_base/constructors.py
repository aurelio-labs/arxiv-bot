import os
import re
import arxiv
import PyPDF2
import json
import requests
from requests.adapters import HTTPAdapter, Retry
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from tqdm.auto import tqdm

from arxiv_bot.utils.logger import logger
from typing import Union, Any, Optional

paper_id_re = re.compile(r'https://arxiv.org/abs/(\d+\.\d+)')

def retry_request_session(retries: Optional[int] = 5):
    # we setup retry strategy to retry on common errors
    retries = Retry(
        total=retries,
        backoff_factor=0.1,
        status_forcelist=[
            408,  # request timeout
            500,  # internal server error
            502,  # bad gateway
            503,  # service unavailable
            504   # gateway timeout
        ]
    )
    # we setup a session with the retry strategy
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_paper_id(query: str, handle_not_found: bool = True):
    """Get the paper ID from a query.

    :param query: The query to search with
    :type query: str
    :param handle_not_found: Whether to return None if no paper is found,
                             defaults to True
    :type handle_not_found: bool, optional
    :return: The paper ID
    :rtype: str
    """
    special_chars = {
        ":": "%3A",
        "|": "%7C",
        ",": "%2C",
        " ": "+"
    }
    # create a translation table from the special_chars dictionary
    translation_table = query.maketrans(special_chars)
    # use the translate method to replace the special characters
    search_term = query.translate(translation_table)
    # init requests search session
    session = retry_request_session()
    # get the search results
    res = session.get(f"https://www.google.com/search?q={search_term}&sclient=gws-wiz-serp")
    try:
        # extract the paper id
        paper_id = paper_id_re.findall(res.text)[0]
    except IndexError:
        if handle_not_found:
            # if no paper is found, return None
            return None
        else:
            # if no paper is found, raise an error
            raise Exception(f'No paper found for query: {query}')
    return paper_id

def init_extractor(
    template: str,
    openai_api_key: Union[str, None] = None,
    max_tokens: int = 1000,
    chunk_size: int = 300,
    chunk_overlap: int = 40
):
    if openai_api_key is None and 'OPENAI_API_KEY' not in os.environ:
        raise Exception('No OpenAI API key provided')
    openai_api_key = openai_api_key or os.environ['OPENAI_API_KEY']
    # instantiate the OpenAI API wrapper
    llm = OpenAI(
        model_name='gpt-3.5-turbo-instruct',
        openai_api_key=openai_api_key,
        max_tokens=max_tokens,
        temperature=0.0
    )
    # initialize prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=['refs']
    )
    # instantiate the LLMChain extractor model
    extractor = LLMChain(
        prompt=prompt,
        llm=llm
    )
    text_splitter = tiktoken_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return extractor, text_splitter

def tiktoken_splitter(chunk_size=300, chunk_overlap=40):
    tokenizer = tiktoken.get_encoding('p50k_base')
    # create length function
    def len_fn(text):
        tokens = tokenizer.encode(
            text, disallowed_special=()
        )
        return len(tokens)
    # initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len_fn,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter


class Arxiv:
    refs_re = re.compile(r'\n(References|REFERENCES)\n')
    references = []
    template = """You are a master PDF reader and when given a set of references you
    always extract the most important information of the papers. For example, when
    you were given the following references:

    Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E.
    Hinton. 2016. Layer normalization. CoRR ,
    abs/1607.06450.
    Eyal Ben-David, Nadav Oved, and Roi Reichart.
    2021. PADA: A prompt-based autoregressive ap-
    proach for adaptation to unseen domains. CoRR ,
    abs/2102.12206.
    Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie
    Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind
    Neelakantan, Pranav Shyam, Girish Sastry, Amanda
    Askell, Sandhini Agarwal, Ariel Herbert-V oss,
    Gretchen Krueger, Tom Henighan, Rewon Child,
    Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu,
    Clemens Winter, Christopher Hesse, Mark Chen,
    Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
    Chess, Jack Clark, Christopher Berner, Sam Mc-
    Candlish, Alec Radford, Ilya Sutskever, and Dario
    Amodei. 2020. Language models are few-shot learn-
    ers. In Advances in Neural Information Processing
    Systems 33: Annual Conference on Neural Informa-
    tion Processing Systems 2020, NeurIPS 2020, De-
    cember 6-12, 2020, virtual .

    You extract the following:

    Layer normalization | Lei Jimmy Ba, Jamie Ryan Kiros, Geoffrey E. Hinton | 2016
    PADA: A prompt-based autoregressive approach for adaptation to unseen domains | Eyal Ben-David, Nadav Oved, Roi Reichart
    Language models are few-shot learners | Tom B. Brown, et al. | 2020

    In the References below there are many papers. Extract their titles, authors, and years.

    References: {refs}

    Extracted:
    """
    llm = None
    get_id = re.compile(r'(?<=arxiv:)\d{4}.\d{5}')

    def __init__(self, paper_id: str):
        """Object to handle the extraction of an ArXiv paper and its
        relevant information.
        
        :param paper_id: The ID of the paper to extract
        :type paper_id: str
        """
        self.id = paper_id
        self.url = f"https://export.arxiv.org/pdf/{paper_id}.pdf"
        # initialize the requests session
        self.session = requests.Session()
    
    def load(self, save: bool = False):
        """Load the paper from the ArXiv API or from a local file
        if it already exists. Stores the paper's text content and
        meta data in self.content and other attributes.
        
        :param save: Whether to save the paper to a local file,
                     defaults to False
        :type save: bool, optional
        """
        # check if pdf already exists
        if os.path.exists(f'papers/{self.id}.json'):
            logger.info(f'Loading papers/{self.id}.json from file')
            with open(f'papers/{self.id}.json', 'r') as fp:
                attributes = json.loads(fp.read())
            for key, value in attributes.items():
                setattr(self, key, value)
        else:
            res = self.session.get(self.url)
            with open(f'temp.pdf', 'wb') as fp:
                fp.write(res.content)
            # extract text content
            self._convert_pdf_to_text()
            # get meta for PDF
            self._download_meta()
            if save:
                self.save()

    def get_refs(self):
        """Get the references for the paper.

        :param extractor: The LLMChain extractor model
        :type extractor: LLMChain
        :param text_splitter: The text splitter to use
        :type text_splitter: TokenTextSplitter
        :return: The references for the paper
        :rtype: list
        """
        logger.info(f'Extracting references for {self.id}')
        if len(self.references) == 0:
            content = self.content.lower()
            matches = self.get_id.findall(content)
            matches = list(set(matches))
            self.references = [{"id": m} for m in matches]
        logger.info(f'Found {len(self.references)} references')
        return self.references
    
    def _convert_pdf_to_text(self):
        """Convert the PDF to text and store it in the self.content
        attribute.
        """
        text = []
        with open("temp.pdf", 'rb') as f:
            # create a PDF object
            pdf = PyPDF2.PdfReader(f)
            # iterate over every page in the PDF
            for page in range(len(pdf.pages)):
                # get the page object
                page_obj = pdf.pages[page]
                # extract text from the page
                text.append(page_obj.extract_text())
        text = "\n".join(text)
        self.content = text

    def _download_meta(self):
        """Download the meta information for the paper from the
        ArXiv API and store it in the self attributes.
        """
        search = arxiv.Search(
            query=f'id:{self.id}',
            max_results=1,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        result = list(search.results())
        if len(result) == 0:
            raise ValueError(f"No paper found for paper '{self.id}'")
        result = result[0]
        # remove 'v1', 'v2', etc. from the end of the pdf_url
        result.pdf_url = re.sub(r'v\d+$', '', result.pdf_url)
        self.authors = [author.name for author in result.authors]
        self.categories = result.categories
        self.comment = result.comment
        self.journal_ref = result.journal_ref
        self.source = result.pdf_url
        self.primary_category = result.primary_category
        self.published = result.published.strftime('%Y%m%d')
        self.summary = result.summary
        self.title = result.title
        self.updated = result.updated.strftime('%Y%m%d')
        logger.info(f"Downloaded metadata for paper '{self.id}'")

    def save(self):
        """Save the paper to a local JSON file.
        """
        with open(f'papers/{self.id}.json', 'w') as fp:
            json.dump(self.__dict__(), fp, indent=4)

    def save_chunks(
        self,
        include_metadata: bool = True,
        path: str = "chunks"
        ):
        """Save the paper's chunks to a local JSONL file.
        
        :param include_metadata: Whether to include the paper's
                                 metadata in the chunks, defaults
                                 to True
        :type include_metadata: bool, optional
        :param path: The path to save the file to, defaults to "papers"
        :type path: str, optional
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/{self.id}-chunks.jsonl', 'w') as fp:
            for chunk in self.dataset:
                if include_metadata:
                    chunk.update(self.get_meta())
                fp.write(json.dumps(chunk) + '\n')
            logger.info(f"Saved paper to '{path}/{self.id}.jsonl'")
        with open(f"{path}/{self.id}.jsonl", "w") as fp:
            fp.write(json.dumps(self.__dict__()))
    
    def get_meta(self):
        """Returns the meta information for the paper.

        :return: The meta information for the paper
        :rtype: dict
        """
        fields = self.__dict__()
        # drop content field because it's big
        fields.pop('content')
        return fields
    
    def chunker(self, chunk_size=300):
        # clean and split into initial smaller chunks
        clean_paper = self._clean_text(self.content)
        splitter = tiktoken_splitter(chunk_size=chunk_size)
        
        langchain_dataset = []

        paper_chunks = splitter.split_text(clean_paper)
        for i, chunk in enumerate(paper_chunks):
            langchain_dataset.append({
                'doi': self.id,
                'chunk-id': str(i),
                'chunk': chunk
            })
        logger.info(f"Split paper into {len(paper_chunks)} chunks")
        self.dataset = langchain_dataset

    def _clean_text(self, text):
        text = re.sub(r'-\n', '', text)
        return text

    def __dict__(self):
        return {
            'id': self.id,
            'title': self.title,
            'summary': self.summary,
            'source': self.source,
            'authors': self.authors,
            'categories': self.categories,
            'comment': self.comment,
            'journal_ref': self.journal_ref,
            'primary_category': self.primary_category,
            'published': self.published,
            'updated': self.updated,
            'content': self.content,
            'references': self.references
        }
    
    def __repr__(self):
        return f"Arxiv(paper_id='{self.id}')"
    

class ArxivGraphScraper:
    def __init__(
        self,
        paper_id: str,
        levels: int = 3,
        save_location: str = 'chunks',
        verbose: bool = False
    ):
        """Build a graph of papers beginning with the paper_id provided,
        identifying the top papers referenced in that paper, downloading
        those papers and repeating. The number of levels to search is
        controlled by the levels parameter.
        
        :param paper_id: The paper ID to start the graph from
        :type paper_id: str
        :param levels: The number of levels to search, defaults to 3
        :type levels: int, optional
        """
        self.paper_id = paper_id
        self.levels = levels
        self.save_location = save_location
        if not os.path.exists(self.save_location):
            os.mkdir(self.save_location)
        # save objects required for ref extraction
        ids = [paper_id]
        for level in tqdm(range(levels)):
            ids = self._build_papers(ids)

    def _create_paper(self, paper_id: str):
        """Create a paper object from a paper ID.
        
        :param paper_id: The paper ID
        :type paper_id: str
        :return: The paper object
        :rtype: ArxivPaper
        """
        logger.info(f"Loading '{paper_id}'")
        paper = Arxiv(paper_id)
        paper.load()
        paper.get_meta()
        logger.info(f"Getting references for paper '{paper.title}'")
        # get references
        refs = paper.get_refs()
        logger.info(f"Found {len(refs)} references")
        paper.chunker()
        paper.save_chunks(include_metadata=True, path=self.save_location)
        return paper
    
    def _build_papers(self, paper_ids: list):
        """Build a list of referenced papers from a list of paper IDs.
        
        :param paper_ids: The list of paper IDs
        :type paper_ids: list
        :return: The list of referenced papers
        :rtype: list
        """
        ids = []
        for _id in tqdm(paper_ids):
            try:
                paper = self._create_paper(_id)
                ids.extend([r['id'] for r in paper.references])
            except Exception as e:
                logger.warning(e)
        original_ids = set(paper_ids)
        new_ids = set(ids)
        new_ids = list(new_ids - original_ids)
        return new_ids
