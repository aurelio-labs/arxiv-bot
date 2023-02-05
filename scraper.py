import os
import requests
import arxivscraper
import pandas as pd
from tqdm.auto import tqdm
import PyPDF2
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "-s", "--start",
    help="Start date for arXiv papers in format YYYY-MM-DD"
)
argParser.add_argument(
    "-e", "--end",
    help="End date for arXiv papers in format YYYY-MM-DD"
)

args = argParser.parse_args()


class Scraper:
    def __init__(self, date_from, date_until):
        if not os.path.exists('./papers'):
            os.mkdir('./papers')
        scraper = arxivscraper.Scraper(
            category='cs',
            date_from=date_from,
            date_until=date_until,
            filters={'categories': ['cs.CL']}
        )
        output = scraper.scrape()
        # convert to pandas dataframe
        cols = ['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors']
        self.records = pd.DataFrame(output, columns=cols)

    def process(self):
        """Download the PDFs, saving one PDF at a time,
        and then transforming them into text files.
        """
        for _, row in tqdm(self.records.iterrows(), total=len(self.records)):
            self._download_pdf(row)
            self._convert_pdf_to_text(row)

    def _download_pdf(self, row):
        url = f"https://arxiv.org/pdf/{row['id']}.pdf"
        r = requests.get(url)
        with open("temp.pdf", 'wb') as f:
            f.write(r.content)

    def _convert_pdf_to_text(self, row):
        text = ""
        try:
            with open("temp.pdf", 'rb') as f:
                # create a PDF object
                pdf = PyPDF2.PdfReader(f)
                # iterate over every page in the PDF
                for page in range(len(pdf.pages)):
                    # get the page object
                    page_obj = pdf.pages[page]
                    # extract text from the page
                    extract = page_obj.extract_text()
                    # add text to data
                    text += extract
            # save the text to a file
            with open(f"papers/{row['id']}.txt", 'w') as f:
                f.write(text)
        except PyPDF2.errors.PdfReadError:
            print(f"Could not read {row['id']}")

if __name__ == '__main__':
    scraper = Scraper(date_from=args.start, date_until=args.end)
    scraper.process()