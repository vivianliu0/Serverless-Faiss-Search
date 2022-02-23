import json
from transformers import (DPRQuestionEncoder, DPRQuestionEncoderTokenizer)
from transformers.file_utils import is_faiss_available, requires_backends

if is_faiss_available():
    import faiss
    

class QueryEncoder:
    def __init__(self, encoded_query_dir: str = None):
        self.has_model = False
        self.has_encoded_query = False
        if encoded_query_dir:
            self.embedding = self._load_embeddings(encoded_query_dir)
            self.has_encoded_query = True

    def encode(self, query: str):
        return self.embedding[query]

    @staticmethod
    def _load_embeddings(encoded_query_dir):
        df = pd.read_pickle(os.path.join(encoded_query_dir, 'embedding.pkl'))
        return dict(zip(df['text'].tolist(), df['embedding'].tolist()))

class DprQueryEncoder(QueryEncoder):

    def __init__(self, encoder_dir: str = None, tokenizer_name: str = None,
                 encoded_query_dir: str = None, device: str = 'cpu'):
        super().__init__(encoded_query_dir)
        if encoder_dir:
            self.device = device
            self.model = DPRQuestionEncoder.from_pretrained("./model")
            self.model.to(self.device)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("./model")
            self.has_model = True
        if (not self.has_model) and (not self.has_encoded_query):
            raise Exception('Neither query encoder model nor encoded queries provided. Please provide at least one')
    
    def encode(self, query: str):
        if self.has_model:
            input_ids = self.tokenizer(query, return_tensors='pt')
            input_ids.to(self.device)
            embeddings = self.model(input_ids["input_ids"]).pooler_output.detach().cpu().numpy()
            return embeddings.flatten()
        else:
            return super().encode(query)

class PRFDenseSearchResult:
    def __init__(self, docid: str, score: str, vectors: [float]):
        requires_backends(self, "faiss")
        self.docid = docid
        self.score = score
        self.vectors = vectors

class SimpleDenseSearcher:
    """Simple Searcher for dense representation
    Parameters
    ----------
    index_dir : str
        Path to faiss index directory.
    """

    def __init__(self, index_dir: str, query_encoder: str):
        requires_backends(self, "faiss")
        self.query_encoder = DprQueryEncoder(query_encoder)
        self.index, self.docids = self.load_index(index_dir)
        self.dimension = self.index.d
        self.num_docs = self.index.ntotal

        assert self.docids is None or self.num_docs == len(self.docids)

    def search(self, query: str, k: int = 10, threads: int = 1, return_vector: bool = False):
        """Search the collection.
        Parameters
        ----------
        query : query text
        k : int
            Number of hits to return.
        threads : int
            Maximum number of threads to use for intra-query search.
        return_vector : bool
            Return the results with vectors
        Returns
        -------
        Tuple[np.ndarray, List[PRFDenseSearchResult]]]
        """
        if isinstance(query, str):
            emb_q = self.query_encoder.encode(query)
            assert len(emb_q) == self.dimension
            emb_q = emb_q.reshape((1, len(emb_q)))

        faiss.omp_set_num_threads(threads)
        if return_vector:
            distances, indexes, vectors = self.index.search_and_reconstruct(emb_q, k)
            vectors = vectors[0]
            distances = distances.flat
            indexes = indexes.flat
            return emb_q, [PRFDenseSearchResult(self.docids[idx], score, vector)
                           for score, idx, vector in zip(distances, indexes, vectors) if idx != -1]
            
    def load_index(self, index_dir: str):
        index_path = index_dir + '/index'
        docid_path = index_dir + '/docid'
        index = faiss.read_index(index_path)
        docids = self.load_docids(docid_path)
        return index, docids
    
    @staticmethod
    def load_docids(docid_path: str):
        id_f = open(docid_path, 'r')
        docids = [line.rstrip() for line in id_f.readlines()]
        id_f.close()
        return docids


def lambda_handler(event, context):
    query = json.loads(event['body'])['query']
    searcher = SimpleDenseSearcher(
        './temp_index',
        'facebook/dpr-question_encoder-multiset-base'
    )
    q_emb, hit = searcher.search(query, k=1, return_vector=True)
    print(hit[0].docid)

    return {
        'statusCode': 200,
        'body': json.dumps(hit[0].docid)
    }
