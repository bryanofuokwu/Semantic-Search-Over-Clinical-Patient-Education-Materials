"""Service for scraping health data from WebMD and other sources."""
import time
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


class ScraperService:
    """Service for scraping health information from WebMD."""
    
    def __init__(self, base_url: str = "https://www.webmd.com", delay: float = 1.0):
        """
        Initialize the scraper service.
        
        Args:
            base_url: Base URL for WebMD
            delay: Delay between requests in seconds (to be respectful)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        try:
            time.sleep(self.delay)  # Be respectful with rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"[WARNING] Failed to fetch {url}: {type(e).__name__}: {e}")
            return None
    
    def get_health_topics_list(self) -> List[Dict[str, str]]:
        """
        Scrape the A-Z health topics list from WebMD.
        
        Returns:
            List of dictionaries with topic name and URL
        """
        url = f"{self.base_url}/a-to-z-guides/health-topics"
        soup = self._get_page(url)
        
        if not soup:
            return []
        
        topics = []
        
        # Find all topic links - WebMD organizes them alphabetically
        # Look for links in the health topics section
        topic_links = soup.find_all('a', href=True)
        
        for link in topic_links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Filter for health topic links
            if '/a-to-z-guides/' in href and text:
                full_url = urljoin(self.base_url, href)
                topics.append({
                    'name': text,
                    'url': full_url
                })
        
        # Remove duplicates
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic['url'] not in seen:
                seen.add(topic['url'])
                unique_topics.append(topic)
        
        print(f"[INFO] Found {len(unique_topics)} unique health topics")
        return unique_topics
    
    def scrape_topic_page(self, topic_url: str, topic_name: str) -> Optional[Dict]:
        """
        Scrape content from a single health topic page.
        
        Args:
            topic_url: URL of the topic page
            topic_name: Name of the topic/condition
            
        Returns:
            Dictionary with scraped content or None if failed
        """
        soup = self._get_page(topic_url)
        
        if not soup:
            return None
        
        # Extract main content
        # WebMD structure varies, so we'll try multiple selectors
        content_selectors = [
            'article',
            '.article-body',
            '.article-content',
            'main',
            '#article-body'
        ]
        
        content_text = ""
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                # Get all paragraphs
                paragraphs = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
                content_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                if content_text:
                    break
        
        # If no structured content found, get all text
        if not content_text:
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            content_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up the text
        content_text = ' '.join(content_text.split())
        
        if not content_text or len(content_text) < 100:
            print(f"[WARNING] Insufficient content scraped from {topic_name}")
            return None
        
        # Try to extract sections if possible
        # This is a simplified extraction - could be enhanced
        sections = self._extract_sections(soup, content_text)
        
        return {
            'title': topic_name,
            'condition': topic_name,
            'category': self._categorize_condition(topic_name),
            'overview': sections.get('overview', content_text[:500]),
            'symptoms': sections.get('symptoms', ''),
            'causes': sections.get('causes', ''),
            'diagnosis': sections.get('diagnosis', ''),
            'treatment_options': sections.get('treatment', ''),
            'self_care': sections.get('self_care', ''),
            'when_to_seek_help': sections.get('when_to_seek_help', ''),
            'faq': sections.get('faq', ''),  # FAQ section if found
            'reading_level': 'standard',
            'full_content': content_text,
            'url': topic_url
        }
    
    def _extract_sections(self, soup: BeautifulSoup, full_text: str) -> Dict[str, str]:
        """Try to extract structured sections from the page."""
        sections = {}
        
        # Look for common section headings
        headings = soup.find_all(['h2', 'h3'])
        current_section = None
        current_content = []
        
        for heading in headings:
            heading_text = heading.get_text(strip=True).lower()
            
            # Save previous section
            if current_section and current_content:
                sections[current_section] = ' '.join(current_content)
            
            # Determine section type
            if any(keyword in heading_text for keyword in ['overview', 'what is', 'about', 'introduction']):
                current_section = 'overview'
            elif any(keyword in heading_text for keyword in ['symptom', 'sign']):
                current_section = 'symptoms'
            elif any(keyword in heading_text for keyword in ['cause', 'risk factor']):
                current_section = 'causes'
            elif any(keyword in heading_text for keyword in ['diagnos', 'test']):
                current_section = 'diagnosis'
            elif any(keyword in heading_text for keyword in ['treat', 'medication', 'therapy']):
                current_section = 'treatment'
            elif any(keyword in heading_text for keyword in ['self', 'manage', 'lifestyle', 'home']):
                current_section = 'self_care'
            elif any(keyword in heading_text for keyword in ['when to', 'seek', 'emergency', 'call']):
                current_section = 'when_to_seek_help'
            else:
                current_section = None
            
            # Get content after this heading
            current_content = []
            next_elem = heading.find_next_sibling()
            while next_elem and next_elem.name not in ['h1', 'h2', 'h3']:
                if next_elem.name in ['p', 'li']:
                    text = next_elem.get_text(strip=True)
                    if text:
                        current_content.append(text)
                next_elem = next_elem.find_next_sibling()
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = ' '.join(current_content)
        
        return sections
    
    def _categorize_condition(self, condition_name: str) -> str:
        """Categorize a condition based on its name."""
        name_lower = condition_name.lower()
        
        if any(term in name_lower for term in ['mental', 'depression', 'anxiety', 'bipolar', 'ptsd', 'adhd']):
            return 'mental_health'
        elif any(term in name_lower for term in ['cancer', 'tumor', 'carcinoma']):
            return 'cancer'
        elif any(term in name_lower for term in ['heart', 'cardiac', 'cardiovascular', 'hypertension', 'cholesterol']):
            return 'cardiovascular'
        elif any(term in name_lower for term in ['diabetes', 'blood sugar']):
            return 'chronic_condition'
        elif any(term in name_lower for term in ['arthritis', 'joint', 'rheumatoid']):
            return 'musculoskeletal'
        elif any(term in name_lower for term in ['skin', 'dermatitis', 'eczema', 'psoriasis']):
            return 'dermatology'
        else:
            return 'general'
    
    def scrape_all_topics(self, max_topics: Optional[int] = None) -> List[Dict]:
        """
        Scrape all health topics from WebMD.
        
        Args:
            max_topics: Maximum number of topics to scrape (None for all)
            
        Returns:
            List of scraped topic dictionaries
        """
        print("[INFO] Fetching health topics list...")
        topics = self.get_health_topics_list()
        
        if max_topics:
            topics = topics[:max_topics]
        
        print(f"[INFO] Scraping {len(topics)} health topics...")
        scraped_data = []
        
        for i, topic in enumerate(topics, 1):
            print(f"[INFO] Scraping {i}/{len(topics)}: {topic['name']}")
            data = self.scrape_topic_page(topic['url'], topic['name'])
            if data:
                scraped_data.append(data)
            else:
                print(f"[WARNING] Failed to scrape {topic['name']}")
        
        print(f"[INFO] Successfully scraped {len(scraped_data)} topics")
        return scraped_data
    
    def save_to_dataframe(self, scraped_data: List[Dict], output_path: Path) -> pd.DataFrame:
        """
        Convert scraped data to DataFrame matching the expected format.
        
        Args:
            scraped_data: List of scraped topic dictionaries
            output_path: Path to save the parquet file
            
        Returns:
            DataFrame with scraped data
        """
        if not scraped_data:
            raise ValueError("No data to save")
        
        # Ensure all required fields are present
        required_fields = ['overview', 'symptoms', 'causes', 'diagnosis', 'treatment_options', 'self_care', 'when_to_seek_help', 'faq']
        for item in scraped_data:
            for field in required_fields:
                if field not in item or not item[field]:
                    # Use full_content as fallback, or empty string if no content
                    item[field] = item.get('full_content', '')[:200] if item.get('full_content') else ''
        
        df = pd.DataFrame(scraped_data)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        print(f"[INFO] Saved {len(df)} records to {output_path}")
        
        return df

