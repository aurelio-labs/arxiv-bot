import os
import requests
import arxiv
import pandas as pd
from tqdm.auto import tqdm
import PyPDF2
import argparse
import re
import logging
from time import time

logging.basicConfig(
    filename="log.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.ERROR
)

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "-m", "--max",
    help="Max results to return, limit is 300_000",
    type=int
)

args = argParser.parse_args()


class Scraper:
    def __init__(self, max_results):
        if not os.path.exists('./papers'):
            os.mkdir('./papers')
        search = arxiv.Search(
            query='cat:cs.CL',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        # create dataset
        dataset = []
        for result in search.results():
            # remove 'v1', 'v2', etc. from the end of the pdf_url
            result.pdf_url = re.sub(r'v\d+$', '', result.pdf_url)
            dataset.append({
                'authors': result.authors,
                'categories': result.categories,
                'comment': result.comment,
                'doi': result.pdf_url[-10:],
                'journal_ref': result.journal_ref,
                'pdf_url': result.pdf_url,
                'primary_category': result.primary_category,
                'published': result.published,
                'summary': result.summary,
                'title': result.title,
                'updated': result.updated
            })
        # convert to pandas dataframe
        self.records = pd.DataFrame(dataset)

    def process(self):
        """Download the PDFs, saving one PDF at a time,
        and then transforming them into text files.
        """
        i = 0
        for _, row in tqdm(self.records.iterrows(), total=len(self.records)):
            i +=1
            if i < 530: continue
            self._download_pdf(row)
            try:
                self._convert_pdf_to_text(row)
                # crawl-delay on export.arxiv.org/robots.txt is 15 seconds
                time.sleep(15)
            except Exception as e:
                print(f"Could not read {row['doi']}, try again")
                logging.error(f"Could not read {row['doi']}: {e}, try again")
                time.sleep(30)
                try:
                    self._convert_pdf_to_text(row)
                except Exception as e:
                    print(f"Could not read {row['doi']}")
                    logging.error(f"Could not read {row['doi']}: {e}")

    def _download_pdf(self, row):
        export_url = "https://export."+row['pdf_url'].split('://')[-1]
        r = requests.get(export_url)
        with open("temp.pdf", 'wb') as f:
            f.write(r.content)

    def _convert_pdf_to_text(self, row):
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
        # save the text to a file
        with open(f"papers/{row['doi']}.txt", 'w') as f:
            f.write(text)

if __name__ == '__main__':
    scraper = Scraper(max_results=args.max)
    scraper.process()