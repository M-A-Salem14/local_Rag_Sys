import lancedb
import pyarrow as pa
from config import DB_DIR, TABLE_NAME, EMBED_DIM

def get_db():
    return lancedb.connect(str(DB_DIR))

def get_or_create_table(db):
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)

    schema = pa.schema([
        pa.field("id",        pa.string()),
        pa.field("text",      pa.string()),         # raw chunk text
        pa.field("source",    pa.string()),         # filename / URL
        pa.field("chunk_idx", pa.int32()),          # position in doc
        pa.field("vector",    pa.list_(pa.float32(), EMBED_DIM)),
    ])

    table = db.create_table(TABLE_NAME, schema=schema)

    # Full-text search index (Tantivy/BM25 under the hood)
    table.create_fts_index("text", replace=True)

    return table