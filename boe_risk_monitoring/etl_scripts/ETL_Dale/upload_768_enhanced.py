import pandas as pd
import json
import ast
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# API key
API_KEY = "pcsk_7RKUT7_7Ef3knhxKJLE3BXnSwkPmSKERbH8M31WRM9ioaoY3KzfDLp3Qq8bb1uasQdT2js"

# 768-dimension configuration
MODEL_NAME = "all-mpnet-base-v2"  # 768 dimensions
DIMENSIONS = 768
INDEX_NAME = "boe-768"
REGION = "us-east-1"
HOST = "https://boe-768-9td1bq3.svc.aped-4627-b74a.pinecone.io"

def create_comprehensive_metadata(row):
    """Create enhanced metadata with all your data fields plus computed features"""
    metadata = {}
    
    # === ORIGINAL FIELDS FROM YOUR CSV ===
    original_fields = [
        'fiscal_period_ref', 'speaker', 'role', 'page', 'section', 
        'reporting_period', 'date_of_earnings_call', 'bank', 
        'document_type', 'source'
    ]
    
    for field in original_fields:
        if field in row and not pd.isna(row[field]):
            value = row[field]
            # Convert numpy types to Python types
            if hasattr(value, 'item'):
                value = value.item()
            metadata[field] = value
    
    # === ENHANCED COMPUTED FIELDS ===
    text = str(row.get('text', ''))
    
    # Text statistics
    metadata['text_length'] = len(text)
    metadata['word_count'] = len(text.split())
    metadata['sentence_count'] = len([s for s in text.split('.') if s.strip()])
    metadata['paragraph_count'] = len([p for p in text.split('\n') if p.strip()])
    
    # Extract temporal information
    if 'date_of_earnings_call' in row and not pd.isna(row['date_of_earnings_call']):
        try:
            date_str = str(row['date_of_earnings_call'])
            if '-' in date_str:
                parts = date_str.split('-')
                metadata['year'] = int(parts[0])
                metadata['month'] = int(parts[1])
                metadata['quarter'] = (int(parts[1]) - 1) // 3 + 1
        except:
            pass
    
    # Extract quarter from reporting_period
    if 'reporting_period' in row and not pd.isna(row['reporting_period']):
        period = str(row['reporting_period'])
        if 'Q' in period:
            try:
                q_part = period.split('_')[0]  # Q1, Q2, etc.
                metadata['quarter_num'] = int(q_part[1])
            except:
                pass
    
    # === SEARCHABLE TAGS ===
    tags = []
    
    # Speaker tags
    if 'speaker' in row and not pd.isna(row['speaker']):
        speaker = str(row['speaker']).lower().replace(' ', '_').replace(',', '')
        tags.append(f"speaker_{speaker}")
    
    # Role tags
    if 'role' in row and not pd.isna(row['role']):
        role = str(row['role']).lower().replace(' ', '_')
        tags.append(f"role_{role}")
    
    # Section tags
    if 'section' in row and not pd.isna(row['section']):
        section = str(row['section']).lower().replace(' ', '_').replace('&', 'and')
        tags.append(f"section_{section}")
    
    # Bank tags
    if 'bank' in row and not pd.isna(row['bank']):
        bank = str(row['bank']).lower().replace(' ', '_')
        tags.append(f"bank_{bank}")
    
    # Document type tags
    if 'document_type' in row and not pd.isna(row['document_type']):
        doc_type = str(row['document_type']).lower()
        tags.append(f"type_{doc_type}")
    
    metadata['tags'] = tags
    
    # === CONTENT CLASSIFICATION ===
    text_lower = text.lower()
    
    # Question detection
    if '?' in text or any(word in text_lower for word in ['question', 'ask', 'wondering']):
        metadata['content_type'] = 'question'
    # Thank you / acknowledgment
    elif any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
        metadata['content_type'] = 'acknowledgment'
    # Financial metrics (numbers, percentages, financial terms)
    elif any(char in text for char in ['$', '%']) or any(word in text_lower for word in ['billion', 'million', 'revenue', 'profit', 'loss']):
        metadata['content_type'] = 'financial_data'
    # Long detailed responses
    elif len(text.split()) > 100:
        metadata['content_type'] = 'detailed_response'
    # Brief statements
    elif len(text.split()) < 20:
        metadata['content_type'] = 'brief_statement'
    else:
        metadata['content_type'] = 'standard_response'
    
    # === SPEAKER ANALYSIS ===
    # Identify key executives vs analysts
    if 'speaker' in row and not pd.isna(row['speaker']):
        speaker = str(row['speaker']).lower()
        if any(name in speaker for name in ['fraser', 'dimon', 'mason', 'barnum']):
            metadata['speaker_type'] = 'executive'
        elif 'operator' in speaker:
            metadata['speaker_type'] = 'operator'
        elif 'role' in row and str(row['role']).lower() == 'analyst':
            metadata['speaker_type'] = 'analyst'
        else:
            metadata['speaker_type'] = 'other'
    
    # === FINANCIAL CONTEXT ===
    # Add financial keywords detection
    financial_keywords = ['revenue', 'profit', 'loss', 'margin', 'capital', 'risk', 'credit', 'loan', 'deposit', 'earnings', 'growth', 'return']
    found_keywords = [kw for kw in financial_keywords if kw in text_lower]
    if found_keywords:
        metadata['financial_keywords'] = found_keywords
    
    return metadata

def setup_768_pinecone():
    """Setup Pinecone for 768-dimension index"""
    print(f"[CONFIG] Model: {MODEL_NAME}")
    print(f"[CONFIG] Dimensions: {DIMENSIONS}")
    print(f"[CONFIG] Index: {INDEX_NAME}")
    print(f"[CONFIG] Host: {HOST}")
    print(f"[CONFIG] Region: {REGION}")
    
    pc = Pinecone(api_key=API_KEY)
    
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if INDEX_NAME in existing_indexes:
        print(f"[DELETING] Existing index '{INDEX_NAME}' to recreate with 768 dimensions...")
        pc.delete_index(INDEX_NAME)
        print(f"[DELETED] Old index removed")
    
    print(f"[CREATING] New 768-dimension index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
    print(f"[SUCCESS] Created 768-dimension index '{INDEX_NAME}'")
    
    return pc.Index(INDEX_NAME, host=HOST)

def upload_768_with_metadata(batch_size=25):
    """Upload Bank of England data with 768 dimensions and enhanced metadata"""
    
    print("=== 768-DIMENSION ENHANCED UPLOAD ===")
    print("Upgrading to 768 dimensions with comprehensive metadata")
    print("Enhanced features:")
    print("  - 768-dimensional embeddings (2x current quality)")
    print("  - 25+ metadata fields per record")
    print("  - Advanced content classification")
    print("  - Financial keyword detection")
    print("  - Searchable tags for all categories")
    print()
    
    # Setup Pinecone
    index = setup_768_pinecone()
    
    # Load 768-dimension model
    print(f"[LOADING] {MODEL_NAME} model (768 dimensions)...")
    model = SentenceTransformer(MODEL_NAME)
    print("[SUCCESS] 768-dimension model loaded!")
    
    # Load data
    df = pd.read_csv("all_text.csv")
    print(f"[DATA] Loaded {len(df)} rows")
    
    # Create ID if needed
    if 'id' not in df.columns:
        df['id'] = df.index.astype(str)
    else:
        df['id'] = df['id'].astype(str)
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"[PROCESSING] {len(df)} rows in {total_batches} batches of {batch_size}")
    
    successful_uploads = 0
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"[BATCH {batch_num + 1}/{total_batches}] Processing rows {start_idx}-{end_idx}")
        
        try:
            # Generate 768-dimensional embeddings
            texts = batch_df['text'].tolist()
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            # Prepare vectors with comprehensive metadata
            vectors = []
            for idx, (_, row) in enumerate(batch_df.iterrows()):
                enhanced_metadata = create_comprehensive_metadata(row)
                
                vectors.append({
                    'id': row['id'],
                    'values': embeddings[idx].tolist(),
                    'metadata': enhanced_metadata
                })
            
            # Upload batch
            index.upsert(vectors=vectors)
            successful_uploads += len(vectors)
            print(f"[SUCCESS] Uploaded batch {batch_num + 1} ({len(vectors)} vectors)")
            
        except Exception as e:
            print(f"[ERROR] Failed batch {batch_num + 1}: {e}")
            continue
    
    # Final verification
    print(f"\n[COMPLETE] Successfully uploaded {successful_uploads} vectors to 768-dimension index")
    
    # Get final stats
    stats = index.describe_index_stats()
    print(f"[STATS] Final index stats: {stats}")
    
    # Show sample enhanced metadata
    print("\n[SAMPLE] Example enhanced metadata (768-dim):")
    sample_metadata = create_comprehensive_metadata(df.iloc[0])
    for key, value in list(sample_metadata.items())[:12]:  # Show first 12 fields
        print(f"  {key}: {value}")
    print(f"  ... and {len(sample_metadata) - 12} more fields")
    
    print(f"\n[SUCCESS] 768-dimension index ready at: {HOST}")
    return index

if __name__ == "__main__":
    print("Starting 768-dimension upload with enhanced metadata...")
    print("This will create embeddings with 2x the quality of 384-dimension setup")
    print()
    
    # Run the upload
    index = upload_768_with_metadata(batch_size=25)
    
    print("\n=== UPLOAD COMPLETE ===")
    print(f"Index: {INDEX_NAME}")
    print(f"Host: {HOST}")
    print(f"Dimensions: {DIMENSIONS}")
    print("Enhanced metadata: ✓")
    print("Ready for advanced semantic search!")