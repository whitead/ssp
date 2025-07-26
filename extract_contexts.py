import os
import json
import re
import time
import datetime
import click
import httpx
import tenacity
import hashlib
import tqdm
import logging

from google import genai
from google.genai import types

from google.cloud import storage



# Configure the genai API key
client = genai.Client(api_key=os.environ['GEMINI_KEY'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EXCTRACT_PROMPT = '''Extract references from the provided paper and format them as JSON according to these specifications:

Output Format:
{
  "$refkey": {
    "title": "string",
    "authors": [
      {
        "family": "string", 
        "given": "string"
      }
    ],
    "year": "string",
    "journal": "string",
    "identifiers": {
      "doi": "string",
      "arxiv": "string"
    },
    "url": "string"
  }
}

Requirements:
1. Use the paper\'s original reference key as $refkey (e.g., "1", "White2024", etc.)
2. Include all available fields from the reference
3. Omit any fields not present in the original reference
4. Preserve exact author name formatting from the paper
5. Include all authors for multi-author papers (omit et al.)
6. Parse DOIs and arXiv IDs into the identifiers object when present

Example Input:
[1] Smith, J.A. and Brown, M.K. (2023). Deep learning approaches in genomics. Nature Methods, 20(3), 234-245. https://doi.org/10.1038/s41592-023-1234-5

Example Output:
{
  "1": {
    "title": "Deep learning approaches in genomics",
    "authors": [
      {
        "family": "Smith",
        "given": "J.A."
      },
      {
        "family": "Brown",
        "given": "M.K."
      }
    ],
    "year": "2023",
    "journal": "Nature Methods",
    "identifiers": {
      "doi": "10.1038/s41592-023-1234-5"
    }
  }
}'''

GET_AUTHOR_PROMPT = '''List the authors of this paper in the following format:

Output Format:
[
  {
    "family": "string",
    "given": "string"
  }
]'''

TOTAL_REFERENCES_PROMPT = '''List the total number of cited references in this paper in this format:

Output Format:
{
  "total_references": "integer"
}'''

def check_author_overlap(n1, n2):
    # Remove initials and check for name intersection
    s1 = {w for w in n1.lower().replace(".", "").split() if len(w) > 1}
    s2 = {w for w in n2.lower().replace(".", "").split() if len(w) > 1}
    return (s1 | s2) == s1 or (s2 | s1) == s2


def sanitize_filename(s):
    """
    Sanitizes a string to be safe for use as a filename by replacing
    non-alphanumeric characters with underscores.
    """
    return re.sub(r'[^A-Za-z0-9_-]', '_', s)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def upload_to_gcp(bucket_name, source_file_path, destination_blob_name):
    storage.Client().bucket(bucket_name).blob(destination_blob_name).upload_from_filename(source_file_path)
    return f"gs://{bucket_name}/{destination_blob_name}"

def get_model_fxn(pdf_path, model, time_minutes=60):
    """Uploads the PDF and returns a model configured with caching."""
    print("Uploading PDF to GCS...")
    # just use md5 for now
    with open(pdf_path, 'rb') as f:
        pdf_md5 = hashlib.md5(f.read()).hexdigest()
    remote_name = os.path.basename(pdf_path).replace(".pdf", f"_{pdf_md5}")
    uri = upload_to_gcp('ad-hoc-fh-internal', pdf_path, f'citation_agreement/{remote_name}.pdf')
    print("Uploaded PDF", uri)
    llm_call = None

    try:
        file_content = [
                types.Content(
                    role='user',
                    parts=[
                        types.Part.from_uri(
                            file_uri=uri,
                            mime_type='application/pdf'),
                    ])
            ]
        cached_content = client.caches.create(
            model=model,
            contents=file_content,
            config=types.CreateCachedContentConfig(
                display_name=os.path.splitext(os.path.basename(pdf_path))[0],
                system_instruction='You are an expert researcher of scholarly papers.',
                ttl=f"{datetime.timedelta(minutes=time_minutes).seconds}s",
            ),
        )
        def llm_call(content):
            response = client.models.generate_content(
                model=model,
                contents=content,
                config=types.GenerateContentConfig(
                    cached_content=cached_content.name,
                    max_output_tokens=8192,
                    response_mime_type= 'application/json',
                    temperature=0
                )
            )
            return response
    except genai.errors.ClientError as e:
        if "400 INVALID_ARGUMENT" in str(e):
            # too small for cache
            def llm_call(content):
                response = client.models.generate_content(
                    model=model,
                    contents=file_content + [content],
                    config=types.GenerateContentConfig(
                        system_instruction='You are an expert researcher of scholarly papers.',
                        temperature= 0,
                        max_output_tokens=8192,
                        response_mime_type= 'application/json',
                    )
                )
                return response

    if llm_call is None:
        raise ValueError("Failed to create cached content for the PDF")

    return llm_call

def parse_llm_json_output(text: str):
    """Extracts JSON data from LLM output."""

    # Use regex to find the JSON content between ```json and ```
    json_matches = re.findall(r'```json(.*?)```', text, re.DOTALL)
    if not json_matches:
        json_content = text
    else:
        json_content = json_matches[0].strip()

    try:
        # Parse the JSON content directly
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        # If parsing fails, provide a detailed error message
        raise ValueError(
            f"Failed to parse JSON. Error: {e}\nJSON Content:\n{json_content}"
        ) from e

@tenacity.retry(
    wait=tenacity.wait_fixed(2),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type((ValueError, TimeoutError)),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
)
def query_model(model, reference):
    """Queries the model for a given reference and parses the JSON response."""
    prompt = (
        f"Find all statements in the main text that cite reference \"{reference}\" for the attached paper. "
        "Report them in the following format:\n\n```json\n[\n  {\n   \"citation_key\": \"key\","
        "\n    \"context\": \"The first sentence(s) of context in which the reference is cited\",\n    \"category\": "
        "\"mention OR support OR contrast\"\n  },\n  {\n    \"reference\": \"key\",\n    "
        "\"context\": \"The second sentence(s) of context in which the reference is cited\",\n    \"category\": "
        "\"mentioned OR supporting OR contrasting\"\n  },\n  ...\n]\n```\n\nwhere \"citation_key\" "
        f"is the key used in the paper (numeric or authoryear value) for \"{reference}\". For example, 20 (numeric) or Foo1988 (authoryear). "
        "\"category\" is an enum. \"support\" means that "
        "the sentence(s) is supported by the citation. \"contrast\" means that the sentence(s) "
        "contrast or disagree with the citation. \"mention\" is if neither apply. The context "
        "is text from the paper that contains the citation. There "
        "should be enough text to understand the reason for the citation without needing "
        f"to have the full text of the paper - but it must cite the considered reference: {reference}."
        "\n\nIf the reference is not mentioned in the text (e.g., the citation to it is found in "
        f"a table), then return an empty JSON list.\n\nDo this only for reference \"{reference}\"."
    )

    # Query the model
    response = model(prompt)

    # Parse the JSON output
    parsed_response = parse_llm_json_output(response.text)
    return parsed_response

def fetch_metadata(doi):
    import os
    
    SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    headers = {'x-api-key': SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    
    with httpx.Client() as client:
        response = client.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "title,authors,references.authors,references.abstract,references.title,references.journal,references.venue,references.year,references.citationStyles,references.externalIds"},
            headers=headers
        )
        response.raise_for_status()
        paper = response.json()
        
        if not paper.get('references'):
            print(paper)
            raise ValueError("Could not find references from Semantic Scholar")
            
        print(f"Ensure title is {paper['title']}!")
        authors = {a['name'] for a in paper['authors']}
        
        # Convert paper authors to the expected format
        formatted_authors = [
            {
                "family": a["name"].split()[-1], # Last name
                "given": " ".join(a["name"].split()[:-1]) # Everything before last name
            }
            for a in paper['authors']
        ]
        
        print(f"Parsing {len(paper['references'])} references")
        # Create references dict in the same format as extract_references
        references = {}
        for i, ref in enumerate(paper['references']):
            key = f'ref-{i}'
            ref_authors = {a['name'] for a in ref['authors']}
            is_self = any(any(check_author_overlap(author, ra) for ra in ref_authors) for author in authors)
            
            # Convert reference authors to the expected format
            formatted_ref_authors = [
                {
                    "family": a["name"].split()[-1],
                    "given": " ".join(a["name"].split()[:-1])
                }
                for a in ref['authors']
            ]
            
            ref_authors_string = ", ".join(a['name'] for a in ref['authors'])
            venue_or_journal = ref.get('venue', '')
            if ref.get('journal', {}) and ref.get('journal').get('name'):
                venue_or_journal = ref['journal']['name']
                
            references[key] = {
                'title': ref.get('title', ''),
                'authors': formatted_ref_authors,
                'year': str(ref.get('year', '')),
                'journal': venue_or_journal,
                'identifiers': {
                    # externalIDs is sometimes explicitly None
                    'doi': (ref.get('externalIds', {}) or {}).get('DOI', ''),
                    'arxiv': (ref.get('externalIds', {}) or {}).get('ArXiv', '')
                },
                'url': '', # S2 API doesn't provide direct URLs
                'key': key,
                'self': is_self,
                'citation': f"{ref.get('title', '')}. {ref_authors_string}. {venue_or_journal}."
            }
            
        print(f"Fraction of self-citations: {sum(r['self'] for r in references.values()) / len(references)}")
        print()
        
        return {
            "references": references,
            "authors": formatted_authors
        }
    

def extract_references(model):

    response = model(GET_AUTHOR_PROMPT)

    author_response = parse_llm_json_output(response.text)
    print(author_response)


    response = model(TOTAL_REFERENCES_PROMPT)
    total_references = int(parse_llm_json_output(response.text)['total_references'])

    all_responses = {}    
    response = model(EXCTRACT_PROMPT + "\n\nOnly complete up to the first 25 references.")
    parsed_response = parse_llm_json_output(response.text)
    all_responses.update(parsed_response)

    # 20, because the models can only count so good
    while len(all_responses) < total_references:
        print(f"Found {len(parsed_response)} references. {total_references - len(all_responses)} remaining. Querying for more...")
        current_list = list(all_responses.keys())
        print(current_list)
        response = model(EXCTRACT_PROMPT + 
                         f"\n\nContinue from this current progress (without repeating this progress):\n\n{json.dumps(all_responses, indent=2)}\n\n"
                         + f"Complete up to the next {min(25, total_references - len(all_responses))} references.")
        parsed_response = parse_llm_json_output(response.text)
        count = len(all_responses)
        all_responses.update(parsed_response)
        print("Added ", len(all_responses) - count)
        if len(all_responses) - count == 0:
            # no progress
            break

    # now build references in expected format
    references = {}
    for key, ref in all_responses.items():
        if 'authors' not in ref:
            continue
        is_self = False
        # drop authors without a family name
        ref['authors'] = [a for a in ref['authors'] if 'family' in a]
        for author in author_response:
            if any(check_author_overlap(author['family'], a['family']) for a in ref['authors']):
                is_self = True
                break
        try:
            ref_authors_string = ", ".join([f"{a['family']}, {a['given']}" for a in ref['authors']])
        except KeyError:
            ref_authors_string = ", ".join([a['family'] for a in ref['authors']])
        references[key] = {
            'title': ref.get('title', ''),
            'authors': ref['authors'],
            'year': ref.get('year', ''),
            'journal': ref.get('journal', ''),
            'identifiers': ref.get('identifiers', {}),
            'url': ref.get('url', ''),
            'key': key,
            'self': is_self
        }
        references[key]['citation'] = f"{references[key]['title']}. {ref_authors_string}. {references[key]['journal']}."

    print(f"Fraction of self-citations: {sum(r['self'] for r in references.values()) / len(references)}")
    print()

    return {"references": references, "authors": author_response}

class CacheManager:
    def __init__(self, doi, cache_path=None):
        self.doi = doi
        name = doi or 'unknown'
        if cache_path:
            self.cache_file = cache_path
        else:
            self.cache_file = f'cache_{sanitize_filename(name)}.json'
        self.metadata = None
        self.results_cache = {}
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                try:
                    cache_data = json.load(f)
                    self.metadata = cache_data.get('metadata')
                    self.results_cache = cache_data.get('results_cache', {})
                    print(f"Loaded metadata and cached query results from '{self.cache_file}'.")
                except json.JSONDecodeError:
                    print("Cache file is corrupted. Fetching new metadata and resetting cache.")
        else:
            print(f"No cache file found for DOI '{self.doi}'.")

    def save_cache(self):
        cache_data = {
            'metadata': self.metadata,
            'results_cache': self.results_cache
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Cache updated and saved to '{self.cache_file}'")

def query_all_references(model, references, cache_manager):
    """Iterates through all references, querying the model and caching results."""
    try:
        with tqdm.tqdm(references.items(), desc="Processing References", ncols=100, bar_format='{l_bar}{bar} {postfix}') as pbar:
            for key, citation in pbar:
                if key in cache_manager.results_cache:
                    pbar.set_postfix_str(f"Skipping cached {key} 😎")
                    continue
                pbar.set_postfix_str(f"Querying {key} 🔍")
                try:
                    parsed_response = query_model(model, citation)
                except Exception as e:
                    pbar.set_postfix_str(f"Error querying {key} ❌")
                    logger.error(f"Error querying reference '{key}': {e}")
                    continue
                cache_manager.results_cache[key] = parsed_response
                time.sleep(1)  # To avoid hitting rate limits
    finally:
        # Save cache regardless of how the loop exits
        cache_manager.save_cache()

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--doi', type=str, default=None, help='DOI of the paper to analyze.')
@click.option('--output-file', type=click.Path(), default='results.json', help='Path to the output JSON file.')
@click.option('--cache-path', type=click.Path(), default=None, help='Path to the cache file.')
def main(pdf_path, doi, output_file, cache_path):
    """CLI tool to query a model over a list of references extracted from a PDF.

    The cache file is automatically named based on the DOI to ensure uniqueness and consistency.
    """
    # Initialize cache manager
    cache_manager = CacheManager(cache_path=cache_path, doi=doi)
    model_name = 'gemini-1.5-pro-002'
    model = get_model_fxn(pdf_path, model_name)


    # If metadata is not loaded from cache, fetch it
    if cache_manager.metadata is None:
        if doi:
            try:
                cache_manager.metadata = fetch_metadata(doi)
            except Exception as e:
                print(f"Error fetching metadata: {e}")
                print("Using extracted references instead.")
                cache_manager.metadata = extract_references(model)
        else:
            cache_manager.metadata = extract_references(model)
        cache_manager.save_cache()

    # Upload PDF and get model

    # Extract references
    references = cache_manager.metadata['references']

    # Query all references
    query_all_references(model, references, cache_manager)

    # Merge results into metadata
    for key, ref in cache_manager.metadata['references'].items():
        if key in cache_manager.results_cache:
            ref['contexts'] = cache_manager.results_cache[key]

    # Save the final results to the output file
    with open(output_file, 'w') as f:
        json.dump(cache_manager.metadata, f, indent=2)
    print(f"Results saved to '{output_file}'")

if __name__ == '__main__':
    main()
