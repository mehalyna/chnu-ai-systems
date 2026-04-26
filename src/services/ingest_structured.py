"""
Structured ingestion for teacher profiles and academic content.
Preserves semantic relationships between entities like teacher names, 
descriptions, and courses.
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from dotenv import load_dotenv
from pathlib import Path
import re
from typing import List, Dict, Optional

load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
USER_AGENT = os.getenv("USER_AGENT")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = os.path.join(BASE_DIR, VECTOR_DB_PATH) if VECTOR_DB_PATH else os.path.join(BASE_DIR, "vector_store")


def parse_teacher_profile(soup: BeautifulSoup, url: str) -> Optional[Dict]:
    """
    Parse a teacher profile page to extract structured information.
    
    Returns:
        Dict with teacher_name, description, courses, email, etc.
    """
    profile = {
        "teacher_name": None,
        "description": "",
        "courses": [],
        "email": None,
        "phone": None,
        "position": None,
        "department": None,
        "url": url
    }
    
    # Try to find teacher name (common patterns)
    name_selectors = [
        'h1', 'h2.teacher-name', '.profile-name', 
        '.teacher-title', '[class*="name"]'
    ]
    
    for selector in name_selectors:
        name_elem = soup.select_one(selector)
        if name_elem and name_elem.get_text(strip=True):
            profile["teacher_name"] = name_elem.get_text(strip=True)
            break
    
    # Extract email
    email_link = soup.find('a', href=re.compile(r'mailto:'))
    if email_link:
        profile["email"] = email_link.get('href', '').replace('mailto:', '')
    
    # Extract phone
    phone_patterns = [r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}']
    text_content = soup.get_text()
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text_content)
        if phone_match:
            profile["phone"] = phone_match.group(0)
            break
    
    # Extract courses - look specifically for teaching sections
    # These keywords indicate actual teaching/course sections
    course_header_keywords = [
        'забезпечує викладання', 'викладає дисципл', 'викладає курс',
        'teaching', 'teaches courses', 'teaches subjects'
    ]
    
    # These keywords indicate publication/research sections (NOT courses)
    publication_keywords = [
        'методичн', 'посібник', 'публікац', 'матеріал', 
        'конференц', 'стаття', 'праці', 'робот',
        'methodical', 'publication', 'conference', 'article', 'paper'
    ]
    
    # These patterns indicate co-authored works (publications, not courses)
    coauthor_indicators = [
        'спільно з', 'разом з', 'методичні рекомендації',
        'jointly with', 'together with', 'co-author'
    ]
    
    # Check for course lists
    lists = soup.find_all(['ul', 'ol'])
    for lst in lists:
        # Check if list is near course-related text
        prev_elem = lst.find_previous(['h2', 'h3', 'h4', 'p', 'strong'])
        if not prev_elem:
            continue
        
        prev_text = prev_elem.get_text().lower()
        
        # Skip if this is a publication section
        if any(keyword in prev_text for keyword in publication_keywords):
            continue
        
        # Only extract if it's clearly a teaching/courses section
        if any(keyword in prev_text for keyword in course_header_keywords):
            for item in lst.find_all('li'):
                course_text = item.get_text(strip=True)
                
                # Filter out publications (they have co-authors, methodical notes, etc.)
                if any(indicator in course_text.lower() for indicator in coauthor_indicators):
                    continue
                
                # Also skip if it contains publication indicators in the text itself
                if any(pub_kw in course_text.lower() for pub_kw in ['методичн', 'посібник', 'стаття']):
                    continue
                
                if course_text and len(course_text) > 3:
                    profile["courses"].append(course_text)
    
    # Extract position/title
    position_keywords = ['професор', 'доцент', 'викладач', 'assistant', 'professor', 'lecturer']
    paragraphs = soup.find_all('p')
    for p in paragraphs[:5]:  # Check first few paragraphs
        text = p.get_text().lower()
        if any(keyword in text for keyword in position_keywords):
            profile["position"] = p.get_text(strip=True)
            break
    
    # Extract main description (paragraphs that don't match other fields)
    description_parts = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if len(text) > 50 and text != profile.get("position"):
            description_parts.append(text)
    
    profile["description"] = "\n\n".join(description_parts[:5])  # Limit to first 5 paragraphs
    
    return profile if profile["teacher_name"] or profile["description"] else None


def create_teacher_document(profile: Dict) -> Document:
    """
    Create a single comprehensive document for a teacher profile.
    All related information stays together in one chunk.
    """
    # Build comprehensive content that keeps everything together
    content_parts = []
    
    if profile["teacher_name"]:
        content_parts.append(f"Викладач: {profile['teacher_name']}")
    
    if profile["position"]:
        content_parts.append(f"Посада: {profile['position']}")
    
    if profile["department"]:
        content_parts.append(f"Кафедра: {profile['department']}")
    
    if profile["email"]:
        content_parts.append(f"Email: {profile['email']}")
    
    if profile["phone"]:
        content_parts.append(f"Телефон: {profile['phone']}")
    
    if profile["description"]:
        content_parts.append(f"\nОпис:\n{profile['description']}")
    
    if profile["courses"]:
        courses_text = "\n• ".join(profile["courses"])
        content_parts.append(f"\nКурси які викладає:\n• {courses_text}")
    
    # Create one unified document
    full_content = "\n\n".join(content_parts)
    
    # Rich metadata for filtering and retrieval
    metadata = {
        "source": profile["url"],
        "type": "teacher_profile",
        "teacher_name": profile["teacher_name"] or "Unknown",
        "email": profile["email"] or "",
        "position": profile["position"] or "",
        "courses": "|".join(profile["courses"]),  # Store as pipe-separated
        "num_courses": len(profile["courses"])
    }
    
    return Document(page_content=full_content, metadata=metadata)


def create_course_documents(profile: Dict) -> List[Document]:
    """
    Create additional documents for each course, linked to the teacher.
    This allows searching by course name to find the teacher.
    """
    course_docs = []
    
    if not profile["courses"] or not profile["teacher_name"]:
        return course_docs
    
    for course in profile["courses"]:
        content = f"""Курс: {course}

Викладач: {profile['teacher_name']}
{f"Посада: {profile['position']}" if profile['position'] else ""}
{f"Email: {profile['email']}" if profile['email'] else ""}
{f"Телефон: {profile['phone']}" if profile['phone'] else ""}

{profile['description'][:300] if profile['description'] else ""}"""
        
        metadata = {
            "source": profile["url"],
            "type": "course_info",
            "course_name": course,
            "teacher_name": profile["teacher_name"],
            "email": profile["email"] or "",
        }
        
        course_docs.append(Document(page_content=content, metadata=metadata))
    
    return course_docs


def ingest_structured_web_content(urls: list[str], topic: str) -> FAISS:
    """
    Ingest web content with structure preservation.
    Keeps teacher profiles intact instead of splitting blindly.
    """
    all_documents = []
    
    for url in urls:
        try:
            print(f"Processing: {url}")
            response = requests.get(url, headers={"User-Agent": USER_AGENT})
            soup = BeautifulSoup(response.content.decode('utf-8', errors='ignore'), "html.parser")
            
            # Try to parse as teacher profile
            profile = parse_teacher_profile(soup, url)
            
            if profile and profile["teacher_name"]:
                # Create main teacher document (keeps everything together)
                teacher_doc = create_teacher_document(profile)
                all_documents.append(teacher_doc)
                print(f"  ✓ Added teacher profile: {profile['teacher_name']}")
                
                # Create course-specific documents for better findability
                course_docs = create_course_documents(profile)
                all_documents.extend(course_docs)
                if course_docs:
                    print(f"  ✓ Added {len(course_docs)} course documents")
            else:
                # Fallback: treat as general content page
                print(f"  ⚠ Not a teacher profile, using general chunking")
                loader = WebBaseLoader(url)
                data = loader.load()
                
                # Use larger chunks for general content
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                chunks = text_splitter.split_documents(data)
                
                # Add context to each chunk
                page_title = url.split('/')[-1].replace('-', ' ').replace('.html', '')
                for doc in chunks:
                    doc.page_content = f"[{page_title}]\n\n{doc.page_content}"
                    doc.metadata["source"] = url
                    doc.metadata["type"] = "general_content"
                
                all_documents.extend(chunks)
                print(f"  ✓ Added {len(chunks)} general chunks")
                
        except Exception as e:
            print(f"  ✗ Error processing {url}: {e}")
            continue
    
    if not all_documents:
        raise ValueError("No documents were successfully processed!")
    
    print(f"\n{'='*60}")
    print(f"Total documents to index: {len(all_documents)}")
    print(f"{'='*60}\n")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Batch ingestion
    batch_size = 100
    vector_store: Optional[FAISS] = None
    
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        
        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
        
        print(f"Indexed {min(i + batch_size, len(all_documents))} / {len(all_documents)} documents...")
    
    if vector_store is None:
        raise ValueError("Failed to create vector store - no documents were indexed")
    
    # Save
    db_path = VECTOR_DB_PATH if VECTOR_DB_PATH else os.path.join(BASE_DIR, "vector_store")
    save_path = os.path.join(db_path, topic)
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"\n✅ Vector store saved to: {save_path}")
    
    return vector_store


def initialize_structured_ingestion(urls: list[str], topic: str):
    """
    Initialize ingestion with structure preservation.
    Entry point for the ingestion API.
    """
    # Fetch all related pages
    all_links = fetch_and_parse_links(urls, depth=10)
    
    # Ingest with structure preservation
    vector_store = ingest_structured_web_content(all_links, topic)
    
    return vector_store


def fetch_and_parse_links(urls: list[str], depth: int) -> list[str]:
    """
    Recursively fetch and parse links from given URLs.
    (Same as original implementation)
    """
    parsed = set()
    links = urls.copy()
    base_url = urls[0] if urls else ""
    
    # Extract base domain for filtering
    from urllib.parse import urlparse
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    for iteration in range(depth):
        new_links = []
        for link in list(links):  # Create a copy to iterate
            if link not in parsed:
                try:
                    response = requests.get(link, headers={"User-Agent": USER_AGENT}, timeout=10)
                    soup = BeautifulSoup(response.content.decode('utf-8', errors='ignore'), "html.parser")
                    
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        
                        # Normalize relative URLs
                        if href.startswith("/"):
                            href = base_domain + href
                        elif not href.startswith("http"):
                            href = base_domain + "/" + href
                        
                        # Filter out non-HTML files and special protocols
                        excluded_extensions = [
                            'pdf', 'jpg', 'png', 'jpeg', 'gif', 'webp', 'svg',
                            'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx',
                            'mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'zip', 'rar'
                        ]
                        excluded_protocols = [
                            'mailto:', 'tel:', 'javascript:', 'ftp:', '#'
                        ]
                        
                        if (href.startswith(base_domain) and 
                            href not in parsed and 
                            not any(ext in href.lower() for ext in excluded_extensions) and
                            not any(proto in href.lower() for proto in excluded_protocols)):
                            new_links.append(href)
                
                except Exception as e:
                    print(f"Error fetching {link}: {e}")
                
                parsed.add(link)
        
        links.extend(new_links)
        links = list(set(links))  # Remove duplicates
        
        print(f"Iteration {iteration + 1}: Found {len(links)} total links")
    
    print(f"\nFinal link count: {len(links)}")
    return links
